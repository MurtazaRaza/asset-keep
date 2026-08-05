"""One query bar, one grammar, compiled to SQL.

```
dark cave                  bare words: FTS over title, notes, filename, tags,
                           plus a semantic pass when CLIP is present
"exact phrase"             FTS phrase
tag:tileset  -tag:wip      include / exclude
kind:image | model3d | audio | reference
license:cc0  source:kenney  root:prototype  collection:jam
w:>=512  h:<64  size:>1mb  tris:<5000  dur:<2s
rate:>48000  channels:1  depth:24  bitrate:>192kbps  peak:>-6db  rms:<-30db
has:alpha | animation | caption | license
is:managed | missing | untagged | vendor | clipping | silent | mono | stereo
similar:1234               CLIP neighbours, or dHash ones
sort:added | name | size | relevance
```

Every filter compiles to an ``EXISTS`` or a comparison rather than a ``JOIN``,
which is what keeps two tag filters from multiplying rows against each other -
the classic way a tag search returns each asset four times.

The parse is deliberately forgiving. An unrecognised prefix is not a syntax
error; ``foo:bar`` becomes two search words. A query bar that rejects input is a
query bar people stop typing into, and there is no case where refusing to search
is more useful than searching for the thing they typed.

The semantic pass is injected rather than imported. :func:`compile_query` takes
a callable that turns a string into ranked asset ids and knows nothing else
about it, so this module never imports onnxruntime, the ranking is testable with
six lines of fake, and a machine with no model installed runs the code that
shipped in M1 rather than a version of it with the interesting parts disabled.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Sequence

from . import similarity, vectors
from .probe import audio
from .tagging import vocab

#: Grammar prefix -> the attribute key the probes actually write.
NUMERIC_FIELDS = {
    "w": "width",
    "width": "width",
    "h": "height",
    "height": "height",
    "size": "bytes",
    "bytes": "bytes",
    "tris": "triangles",
    "triangles": "triangles",
    "verts": "vertices",
    "dur": "duration",
    "duration": "duration",
    "frames": "frame_count",
    "cols": "cols",
    "rows": "rows",
    "colors": "color_count",
    # Audio. `rate` and `channels` are the two that answer a real import
    # question in Unity - a 96 kHz sound effect is wasted bytes, and a stereo
    # one cannot be positioned in 3D - and both were probed from M1 without ever
    # being reachable from the query bar.
    "rate": "sample_rate",
    "samplerate": "sample_rate",
    "channels": "channels",
    "depth": "bit_depth",
    "bitrate": "bitrate",
    "peak": "peak_db",
    "rms": "rms_db",
}

#: ``has:x`` -> the attribute that has to be truthy.
HAS_FIELDS = {
    "alpha": "has_alpha",
    "animation": "animation_count",
    "rig": "has_rig",
    "frames": "frame_count",
}

SORTS = {
    "added": "a.added_at DESC",
    "updated": "a.updated_at DESC",
    "name": "a.title COLLATE NOCASE ASC",
    "size": "(SELECT MAX(size) FROM location WHERE asset_id = a.id) DESC",
    "relevance": "score ASC",  # bm25 returns smaller for better matches
}

DEFAULT_SORT = "added"

#: How many dHash neighbours ``similar:`` returns, nearest first.
SIMILAR_LIMIT = 60

#: Differing bits above which a match is no better than chance. Two random
#: 64-bit hashes differ in 32 bits on average, so this excludes noise and
#: nothing else - which is deliberate.
#:
#: The tempting design is a tight quality threshold, and it does not survive
#: contact with a real library. Measured across the calibration project, the
#: distances from one sprite to every other run 13, 17, 18, 18, 19, 19, ... 21,
#: 23, 25, with no gap anywhere: the nearest genuine match sits at 13 and
#: unrelated art starts blending in around 25. A cutoff of 12 - which reads
#: entirely sensible, and is the figure usually quoted - returns nothing at all
#: for most assets. The distribution is smooth because these are sprites and
#: tiling textures, not photographs, so "similar" here is a ranking question
#: rather than a membership one. Rank, cap the count, and let the eye decide.
SIMILAR_MAX_DISTANCE = 32

#: How many semantic hits a bare-word query may add to its literal ones.
#:
#: A cap is not a tuning knob here, it is the definition. FTS answers a yes/no
#: question and returns a set; a cosine returns a number for every asset in the
#: library and never says no. So the semantic side has to be "the best N", and
#: N is chosen to be worth paging through and no larger.
SEMANTIC_LIMIT = 120

#: Image-text cosine below which a semantic hit is not a hit. A third of
#: :data:`assetkeep.vectors.NEIGHBOUR_FLOOR`, because image-image and image-text
#: scores are not on the same scale at all: the two towers put their outputs in
#: different regions of the space, so text never scores against an image the way
#: another image does, and a number that is obviously low for one is obviously
#: high for the other.
#:
#: Measured over fourteen realistic queries against the calibration library, the
#: pooled image-text distribution has a median of 0.223 and a 99th percentile of
#: 0.271. So 0.24 is roughly "the top one percent": it leaves a median of 62
#: results per query and no query empty-handed, where 0.28 leaves half of them
#: with nothing at all.
SEMANTIC_FLOOR = 0.24

#: Reciprocal-rank fusion's damping constant, from the paper that introduced it.
#: Large enough that one list's confident first place cannot outvote several
#: agreeing mid-list results, which is the failure this is here to prevent.
RRF_K = 60

_UNIT_MULTIPLIERS = {
    "": 1, "b": 1, "kb": 1024, "k": 1024, "mb": 1024**2, "m": 1024**2,
    "gb": 1024**3, "g": 1024**3, "s": 1, "ms": 0.001,
    # Decibels are already a bare number; the unit exists so that `peak:>-6db`
    # reads the way a person would write it. Bitrates are the one place the
    # binary `k` above is the wrong prefix - a 192 kbps file is 192,000 bits per
    # second, not 196,608 - so `kbps` is decimal on purpose.
    "db": 1, "khz": 1000, "kbps": 1000,
}

_TOKEN = re.compile(r'-?(?:[a-zA-Z_]+:)?"[^"]*"|\S+')
_COMPARISON = re.compile(r"^(>=|<=|>|<|=)?(.*)$")
#: The leading minus is for decibels, which are the only negative numbers in the
#: grammar. It cannot collide with the `-tag:wip` negation, which is stripped
#: from the front of the whole token well before a value is parsed.
_NUMBER = re.compile(r"^(-?[0-9]*\.?[0-9]+)\s*([a-z]*)$", re.IGNORECASE)


@dataclass(frozen=True)
class Term:
    field: str
    operator: str
    value: str
    negated: bool = False


@dataclass(frozen=True)
class Query:
    terms: tuple[Term, ...] = ()
    #: Bare words and quoted phrases, for FTS.
    text: tuple[str, ...] = ()
    sort: str = DEFAULT_SORT
    similar_to: int | None = None
    raw: str = ""
    #: Whether ``sort:`` was typed, as opposed to defaulted. The semantic pass
    #: needs to tell those apart: fusing two rankings and then throwing the
    #: result away to sort by date is not a merge, it is a slower filter, but
    #: somebody who explicitly asked for ``sort:name`` meant it.
    sort_explicit: bool = False

    def with_term(self, term: Term) -> "Query":
        """Add one filter, which is how a sidebar click edits the query."""
        added = Query(
            terms=self.terms + (term,),
            text=self.text,
            sort=self.sort,
            similar_to=self.similar_to,
            sort_explicit=self.sort_explicit,
        )
        return Query(
            terms=added.terms,
            text=added.text,
            sort=added.sort,
            similar_to=added.similar_to,
            raw=serialise(added),
            sort_explicit=added.sort_explicit,
        )


def parse(text: str) -> Query:
    """Turn a query string into a filter tree.

    >>> q = parse('dark cave tag:tileset -tag:wip w:>=512 sort:name')
    >>> q.text
    ('dark', 'cave')
    >>> [(t.field, t.operator, t.value, t.negated) for t in q.terms]
    [('tag', '=', 'tileset', False), ('tag', '=', 'wip', True), ('w', '>=', '512', False)]
    >>> q.sort
    'name'
    >>> parse('foo:bar').text  # unknown prefixes are just words
    ('foo:bar',)
    """
    terms: list[Term] = []
    words: list[str] = []
    sort = DEFAULT_SORT
    sort_explicit = False
    similar_to: int | None = None

    for token in _TOKEN.findall(text):
        negated = token.startswith("-")
        body = token[1:] if negated else token

        field_name, separator, value = body.partition(":")
        field_name = field_name.lower()

        if not separator or field_name not in _KNOWN_FIELDS:
            words.append(body.strip('"'))
            continue

        value = value.strip('"')
        if field_name == "sort":
            sort = value if value in SORTS else DEFAULT_SORT
            sort_explicit = value in SORTS
            continue
        if field_name == "similar":
            similar_to = int(value) if value.isdigit() else None
            continue

        operator, operand = _COMPARISON.match(value).groups()
        terms.append(Term(field_name, operator or "=", operand, negated))

    return Query(
        tuple(terms), tuple(words), sort, similar_to, text, sort_explicit
    )


def serialise(query: Query) -> str:
    """Render a query back to its string form, so chips and the bar agree.

    >>> serialise(parse('tag:tileset -tag:wip w:>=512 sort:name dark'))
    'dark tag:tileset -tag:wip w:>=512 sort:name'
    """
    parts = list(query.text)
    for term in query.terms:
        prefix = "-" if term.negated else ""
        operator = "" if term.operator == "=" else term.operator
        parts.append(f"{prefix}{term.field}:{operator}{term.value}")
    if query.similar_to is not None:
        parts.append(f"similar:{query.similar_to}")
    if query.sort != DEFAULT_SORT:
        parts.append(f"sort:{query.sort}")
    return " ".join(parts)


#: A callable turning query text into ranked ``(asset_id, score)`` pairs. Held
#: by the server and the CLI, which have a config and can load a model; passed
#: in here, which has neither.
Semantic = Callable[[str], Sequence[tuple[int, float]]]


def search(
    conn: sqlite3.Connection,
    text: str,
    limit: int = 200,
    offset: int = 0,
    semantic: Semantic | None = None,
) -> list[sqlite3.Row]:
    """Run a query string and return matching asset rows."""
    sql, params = compile_query(conn, parse(text), limit, offset, semantic)
    return conn.execute(sql, params).fetchall()


def compile_query(
    conn: sqlite3.Connection,
    query: Query,
    limit: int = 200,
    offset: int = 0,
    semantic: Semantic | None = None,
) -> tuple[str, list]:
    """Build the SQL for a parsed query, plus its parameters."""
    conditions: list[str] = []
    params: list = []
    params_tail: list = []
    source = "asset a"
    order = SORTS.get(query.sort, SORTS[DEFAULT_SORT])
    ranked: list[int] | None = None

    if query.text and semantic is not None:
        # Two retrievals, fused. The literal side stays exhaustive - every FTS
        # match is still a result, exactly as without a model - and the semantic
        # side adds its best few. What changes is the order, not the guarantee.
        ranked = fuse(
            _literal_ids(conn, query.text),
            [asset_id for asset_id, _ in semantic(" ".join(query.text))],
        )
        if not ranked:
            return "SELECT * FROM asset WHERE 0", []
        conditions.append(f"a.id IN ({','.join('?' * len(ranked))})")
        params.extend(ranked)
    elif query.text:
        # Joining the scored subquery filters and ranks in one pass, and keeps
        # bm25 available for sort:relevance without matching twice.
        source = (
            "asset a JOIN (SELECT rowid, bm25(asset_fts) AS score FROM asset_fts "
            "WHERE asset_fts MATCH ?) m ON m.rowid = a.id"
        )
        params.append(_fts_match(query.text))
        order = order.replace("score", "m.score")
    elif query.sort == "relevance":
        order = SORTS[DEFAULT_SORT]

    if query.similar_to is not None:
        neighbours = similar_ids(conn, query.similar_to)
        if not neighbours:
            return "SELECT * FROM asset WHERE 0", []
        conditions.append(f"a.id IN ({','.join('?' * len(neighbours))})")
        params.extend(neighbours)
        # Nearest first beats any other ordering once you have asked for this,
        # including a fusion: ``similar:412 dark`` is a question about 412.
        order, params_tail = _rank_order(neighbours)
    elif ranked is not None and not (query.sort_explicit and query.sort != "relevance"):
        # The fused rank is the relevance ordering, so it answers both the
        # default and an explicit sort:relevance. Any other explicit sort is
        # someone overriding it on purpose.
        order, params_tail = _rank_order(ranked)

    for term in query.terms:
        clause, values = _compile_term(term)
        if clause is None:
            continue
        conditions.append(f"NOT ({clause})" if term.negated else clause)
        params.extend(values)

    where = " AND ".join(conditions) if conditions else "1"
    sql = (
        f"SELECT a.* FROM {source} WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?"
    )
    return sql, params + params_tail + [limit, offset]


def _literal_ids(conn: sqlite3.Connection, text: tuple[str, ...]) -> list[int]:
    """FTS matches in bm25 order, as ids.

    Pulled out of SQL and back into Python only when there is a second ranking
    to fuse with. Without one the join is strictly better, which is why both
    paths exist rather than the simpler-looking single one.
    """
    return [
        int(row[0])
        for row in conn.execute(
            "SELECT rowid FROM asset_fts WHERE asset_fts MATCH ? "
            "ORDER BY bm25(asset_fts)",
            (_fts_match(text),),
        )
    ]


def _rank_order(ids: Sequence[int]) -> tuple[str, list]:
    """A CASE expression putting rows back into the order they were ranked in."""
    whens = " ".join(f"WHEN ? THEN {position}" for position in range(len(ids)))
    return f"CASE a.id {whens} END", list(ids)


def fuse(*rankings: Sequence[int]) -> list[int]:
    """Reciprocal-rank fusion of several ranked id lists, best first.

    Ranks, not scores, and that is the whole argument. bm25 returns an unbounded
    negative number whose scale depends on the corpus; a cosine returns 0.15 to
    0.35 in a band that depends on the prompt. There is no honest way to put
    those on one axis, and every attempt to - min-max within the page, a fixed
    multiplier, a hand-tuned weight - produces a constant that has to be
    retuned every time either side changes. What both methods genuinely agree
    on is what an ordering is, so fuse the orderings.

    An id appearing in both lists beats one that topped a single list, which is
    the behaviour worth having: the literal and semantic passes have completely
    different failure modes, so the things they agree on are rarely wrong.

    >>> fuse([1, 2, 3], [3, 9])
    [3, 1, 2, 9]
    >>> fuse([], [4, 5])
    [4, 5]
    >>> fuse()
    []
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, asset_id in enumerate(ranking):
            scores[asset_id] = scores.get(asset_id, 0.0) + 1.0 / (RRF_K + position + 1)
    # Ties broken by id so the same query twice is the same page twice.
    return sorted(scores, key=lambda asset_id: (-scores[asset_id], asset_id))


def similar_ids(
    conn: sqlite3.Connection, asset_id: int, model: str | None = None
) -> list[int]:
    """Asset ids most like ``asset_id``, nearest first.

    CLIP's neighbours when that asset has an embedding, the perceptual hash's
    otherwise, and the difference is worth stating: dHash answers "does this
    look like that", embeddings answer "is this the same sort of thing". Two
    goblins drawn by different artists are the second and not the first.

    The fallback is not a degraded mode, it is the M1 behaviour, still exactly
    as good as it was at telling near-duplicates apart - which is the job it was
    always best at.
    """
    model = model or vectors.primary_model(conn)
    if model is not None:
        found = vectors.neighbours(conn, model, asset_id, limit=SIMILAR_LIMIT)
        if found:
            return [found_id for found_id, _ in found]

    row = conn.execute(
        "SELECT dhash FROM phash WHERE asset_id = ?", (asset_id,)
    ).fetchone()
    if row is None:
        return []

    target = similarity.from_signed(int(row["dhash"]))
    scored = [
        (similarity.hamming(target, similarity.from_signed(int(other["dhash"]))), int(other["asset_id"]))
        for other in conn.execute("SELECT asset_id, dhash FROM phash")
        if int(other["asset_id"]) != asset_id
    ]
    scored.sort()
    return [
        found for distance, found in scored[:SIMILAR_LIMIT]
        if distance <= SIMILAR_MAX_DISTANCE
    ]


def facets(
    conn: sqlite3.Connection,
    text: str,
    limit: int = 50,
    semantic: Semantic | None = None,
) -> list[sqlite3.Row]:
    """Tag counts across everything the current query matches.

    Counted over the result set rather than the whole library, so the sidebar
    answers "what else is in here" rather than "what exists somewhere". Given
    the same semantic pass as the grid, for the same reason: a sidebar counting
    a different set from the one on screen is worse than no sidebar.
    """
    inner, params = compile_query(
        conn, parse(text), limit=100_000, offset=0, semantic=semantic
    )
    return conn.execute(
        f"""
        SELECT t.name, t.namespace, COUNT(DISTINCT at.asset_id) AS count
        FROM asset_tag at
        JOIN tag t ON t.id = at.tag_id
        WHERE at.asset_id IN (SELECT id FROM ({inner}))
        GROUP BY t.id ORDER BY count DESC, t.name LIMIT ?
        """,
        params + [limit],
    ).fetchall()


def _compile_term(term: Term) -> tuple[str | None, list]:
    field_name, operator, value = term.field, term.operator, term.value

    if field_name == "tag":
        return (
            "EXISTS (SELECT 1 FROM asset_tag at JOIN tag t ON t.id = at.tag_id "
            "WHERE at.asset_id = a.id AND t.name = ?)",
            [vocab.canonical(value)],
        )

    if field_name == "kind":
        return "a.kind = ?", [value.lower()]

    if field_name in ("license", "source"):
        column = "license" if field_name == "license" else "source_name"
        return f"a.{column} LIKE ?", [f"%{value}%"]

    if field_name == "root":
        return (
            "EXISTS (SELECT 1 FROM location l JOIN root r ON r.id = l.root_id "
            "WHERE l.asset_id = a.id AND (r.name = ? OR r.path LIKE ?))",
            [vocab.canonical(value), f"%{value}%"],
        )

    if field_name == "collection":
        # Canonicalised on both sides, the same as tags and root names, so a
        # sidebar click and a typed filter reach the same collection.
        return (
            "EXISTS (SELECT 1 FROM collection_asset ca JOIN collection c "
            "ON c.id = ca.collection_id WHERE ca.asset_id = a.id AND c.name = ?)",
            [vocab.canonical(value)],
        )

    if field_name in NUMERIC_FIELDS:
        number = _number(value)
        if number is None:
            return None, []
        return (
            "EXISTS (SELECT 1 FROM attribute at2 WHERE at2.asset_id = a.id "
            f"AND at2.key = ? AND at2.value_num {operator} ?)",
            [NUMERIC_FIELDS[field_name], number],
        )

    if field_name == "has":
        return _compile_has(value)

    if field_name == "is":
        return _compile_is(value)

    return None, []


def _compile_has(value: str) -> tuple[str | None, list]:
    value = value.lower()
    if value == "caption":
        return "a.caption IS NOT NULL AND a.caption != ''", []
    if value == "license":
        return "a.license IS NOT NULL AND a.license != ''", []
    if value == "source":
        return "a.source_url IS NOT NULL AND a.source_url != ''", []
    if value in HAS_FIELDS:
        return (
            "EXISTS (SELECT 1 FROM attribute at2 WHERE at2.asset_id = a.id "
            "AND at2.key = ? AND at2.value_num > 0)",
            [HAS_FIELDS[value]],
        )
    return None, []


def _compile_is(value: str) -> tuple[str | None, list]:
    value = value.lower()
    if value == "managed":
        return "a.managed = 1", []
    if value == "missing":
        # References are excluded rather than incidentally matching everything:
        # they have no location by design, and a filter that answered "which
        # files have gone" with "all of your links" would be useless the day
        # somebody adds one.
        return (
            "a.kind != 'reference' AND NOT EXISTS (SELECT 1 FROM location l "
            "WHERE l.asset_id = a.id AND l.present = 1)",
            [],
        )
    if value == "untagged":
        # Structural tags are applied to everything, so "untagged" has to mean
        # "nothing but structure" or it would never match anything at all.
        return (
            "NOT EXISTS (SELECT 1 FROM asset_tag at2 WHERE at2.asset_id = a.id "
            "AND at2.source != 'structural')",
            [],
        )
    if value == "vendor":
        return (
            "EXISTS (SELECT 1 FROM location l JOIN root r ON r.id = l.root_id "
            "WHERE l.asset_id = a.id AND r.vendor = 1)",
            [],
        )

    # Audio levels. These are `peak:` and `channels:` underneath and exist
    # anyway, because the useful queries here are ones nobody will phrase as a
    # number: "which of these is clipping" and "which of these is silent" are
    # the questions, and `peak:>=-0.1` is an answer to a question asked
    # backwards.
    if value == "clipping":
        return _attribute_at_least("peak_db", audio.CLIPPING_DB)
    if value == "silent":
        return _attribute_below("peak_db", audio.SILENT_DB)
    if value in ("mono", "stereo"):
        return (
            "EXISTS (SELECT 1 FROM attribute at2 WHERE at2.asset_id = a.id "
            "AND at2.key = 'channels' AND at2.value_num = ?)",
            [1.0 if value == "mono" else 2.0],
        )
    return None, []


def _attribute_at_least(key: str, threshold: float) -> tuple[str, list]:
    return (
        "EXISTS (SELECT 1 FROM attribute at2 WHERE at2.asset_id = a.id "
        "AND at2.key = ? AND at2.value_num >= ?)",
        [key, threshold],
    )


def _attribute_below(key: str, threshold: float) -> tuple[str, list]:
    return (
        "EXISTS (SELECT 1 FROM attribute at2 WHERE at2.asset_id = a.id "
        "AND at2.key = ? AND at2.value_num < ?)",
        [key, threshold],
    )


def _fts_match(words: tuple[str, ...]) -> str:
    """Build an fts5 MATCH expression that cannot be a syntax error.

    Every word is quoted, so ``NOT``, ``*``, ``(`` and the rest arrive as text.
    A user typing ``AND`` into an asset search means the word, and an fts5
    syntax error would surface as an empty result with no explanation.

    >>> _fts_match(("dark", 'cave "of" NOT'))
    '"dark" "cave ""of"" NOT"'
    """
    return " ".join('"' + word.replace('"', '""') + '"' for word in words if word)


def _number(value: str) -> float | None:
    """Parse ``512``, ``1mb`` or ``2s`` into a plain number.

    >>> _number("512"), _number("1mb"), _number("2s"), _number("250ms")
    (512.0, 1048576.0, 2.0, 0.25)
    >>> _number("wide") is None
    True

    Decibels are negative, and ``kbps`` is decimal where ``kb`` is binary:

    >>> _number("-6db"), _number("192kbps"), _number("192kb")
    (-6.0, 192000.0, 196608.0)
    """
    match = _NUMBER.match(value.strip())
    if match is None:
        return None
    amount, unit = match.groups()
    multiplier = _UNIT_MULTIPLIERS.get(unit.lower())
    if multiplier is None:
        return None
    return float(amount) * multiplier


_KNOWN_FIELDS = (
    {"tag", "kind", "license", "source", "root", "collection", "has", "is",
     "sort", "similar"}
    | set(NUMERIC_FIELDS)
)
