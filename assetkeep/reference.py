"""References: the assets that are a URL rather than a file.

An asset store page, a tutorial, an artist's post you keep going back to. These
are half of what "where did this come from" means in practice, and a library
that can hold the sprite but not the page it came off makes you keep the second
half somewhere else - which in practice means nowhere.

**A reference is an ordinary asset.** Same table, same tags, same collections,
same search grammar, same inspector. ``kind:reference`` is the only thing that
distinguishes one, and it exists so that ``kind:image`` still means what it
says. Everything downstream that was written for files works on references
because there was never a second code path for them to diverge from.

**Identity is the URL, hashed the same way file content is.** ``content_hash``
holds ``blake2b`` of the normalised URL rather than of any bytes, which the
schema comment originally said would be ``NULL``. Three things fall out of the
change and none of them fall out of ``NULL``: adding the same link twice is one
asset, because ``content_hash`` is already ``UNIQUE`` and the deduplication a
scan gets for free is exactly the behaviour wanted here; the thumbnail store is
content-addressed by that column, so an ``og:image`` lands in the same sharded
tree as every other tile with no second mechanism; and the frontend's
``/api/thumb/{hash}`` needs no reference-shaped special case. What it costs is
that ``content_hash`` means "identity" rather than "hash of the bytes", which
is what :mod:`assetkeep.hashing` says it is for anyway.

**Fetching is optional and always survivable.** The URL is the asset. A title,
a description and a preview are what a successful fetch adds, and a dead link
still gets a reference named after its own last path segment - which is usually
the slug, and is usually enough to recognise it by.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from . import db, hashing, thumbs
from .config import Config
from .probe import reference as reference_probe
from .probe.reference import Page
from .tagging import heuristic, vocab

log = logging.getLogger(__name__)

#: Mixed into the digest so a reference's identity is drawn from a different
#: input space than a file's. Both are 128-bit blake2b and a collision between
#: them is not going to happen; keeping them separable is free, and it means the
#: digest of a URL cannot be produced by hashing any file's contents.
IDENTITY_PREFIX = b"assetkeep:url:"


@dataclass
class Reference:
    """What adding or refreshing one URL did."""

    asset_id: int
    url: str
    title: str
    #: False when the URL was already in the library and this updated it.
    created: bool = False
    #: Whether a preview image was fetched and stored.
    thumbnail: bool = False
    #: The fetch's own failure, if it had one. The reference exists either way.
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "url": self.url,
            "title": self.title,
            "created": self.created,
            "thumbnail": self.thumbnail,
            "error": self.error,
        }


def identity(url: str) -> str:
    """The content hash a reference to this URL gets.

    Normalised first, so the same page reached with a fragment or a capitalised
    host is the same asset.

    >>> identity("https://kenney.nl/assets") == identity("kenney.nl/assets#top")
    True
    >>> identity("https://kenney.nl/assets") == identity("https://kenney.nl/other")
    False
    >>> len(identity("https://kenney.nl/assets"))
    32
    """
    normalised = reference_probe.normalise(url) or (url or "").strip()
    return hashing.hash_bytes(IDENTITY_PREFIX + normalised.encode("utf-8"))


def existing(conn: sqlite3.Connection, url: str) -> sqlite3.Row | None:
    """The asset already holding this URL, if there is one."""
    return db.asset_by_hash(conn, identity(url))


def add(
    conn: sqlite3.Connection,
    config: Config,
    url: str,
    title: str | None = None,
    notes: str | None = None,
    license: str | None = None,
    tags=(),
    fetch: bool = True,
) -> Reference:
    """Put a URL in the library, fetching what the page says about itself.

    An explicit ``title`` wins over the fetched one, always. Somebody typing a
    name for a link has said something the page cannot contradict, and the whole
    reason to allow it is that a page's own title is sometimes ``Home``.

    ``fetch=False`` skips the network entirely, which is what makes adding a
    hundred links from a text file a local operation.
    """
    normalised = reference_probe.normalise(url)
    if normalised is None:
        raise ValueError(f"not a fetchable URL: {url!r}")

    page = (
        reference_probe.fetch(normalised)
        if fetch
        else Page(
            url=normalised,
            title=reference_probe.fallback_title(normalised),
            site_name=reference_probe.host_of(normalised),
        )
    )

    digest = identity(normalised)
    known = db.asset_by_hash(conn, digest)
    created = known is None
    if known is None:
        asset_id = db.create_asset(
            conn, "reference", digest, title or page.title or normalised
        )
    else:
        asset_id = int(known["id"])

    result = _record(
        conn,
        config,
        asset_id,
        page,
        title=title,
        notes=notes,
        license=license,
        overwrite=created,
    )
    result.created = created

    if tags:
        db.add_tags(conn, asset_id, [(str(name), "manual", None) for name in tags])
        db.index_fts(conn, asset_id)
    return result


def refresh(
    conn: sqlite3.Connection,
    config: Config,
    asset_id: int,
    overwrite: bool = False,
) -> Reference | None:
    """Fetch a reference's page again. ``None`` if that asset is not one.

    By default this fills in what is missing and leaves what is there alone.
    Renaming a link to something you will recognise is a normal thing to do, and
    a refetch that silently reverts it to ``Untitled Document`` would make this
    button one nobody presses twice. ``overwrite`` is for the other case: the
    page has genuinely changed and its own metadata is now the better one.
    """
    row = conn.execute(
        "SELECT * FROM asset WHERE id = ? AND kind = 'reference'", (asset_id,)
    ).fetchone()
    if row is None:
        return None

    url = row["source_url"] or ""
    if not url:
        return Reference(
            asset_id=asset_id,
            url="",
            title=row["title"],
            error="this reference has no URL to fetch",
        )

    return _record(
        conn, config, asset_id, reference_probe.fetch(url), overwrite=overwrite
    )


def _record(
    conn: sqlite3.Connection,
    config: Config,
    asset_id: int,
    page: Page,
    title: str | None = None,
    notes: str | None = None,
    license: str | None = None,
    overwrite: bool = True,
) -> Reference:
    """Write one fetched page onto an asset, tags, preview and all."""
    current = conn.execute(
        "SELECT * FROM asset WHERE id = ?", (asset_id,)
    ).fetchone()

    fields: dict[str, str] = {"source_url": page.url}
    if title is not None:
        fields["title"] = title
    elif page.title and (overwrite or not (current["title"] or "").strip()):
        fields["title"] = page.title

    if notes is not None:
        fields["notes"] = notes
    elif page.description and (overwrite or not (current["notes"] or "").strip()):
        fields["notes"] = page.description

    if license is not None:
        fields["license"] = license
    if page.site_name and (overwrite or not (current["source_name"] or "").strip()):
        fields["source_name"] = page.site_name

    db.update_asset(conn, asset_id, fields)

    host = reference_probe.host_of(page.url)
    db.set_attributes(
        conn,
        asset_id,
        {
            "host": host,
            "fetched_at": db.now(),
            # Recorded rather than inferred from a missing thumbnail, so the
            # inspector can say "the page did not answer" instead of leaving a
            # blank tile to be read as a bug.
            "fetch_error": page.error,
        },
    )

    _apply_tags(conn, asset_id, fields.get("title", current["title"]), host)

    stored = _store_preview(
        config, page, str(current["content_hash"]), replace=overwrite
    )
    db.index_fts(conn, asset_id)

    return Reference(
        asset_id=asset_id,
        url=page.url,
        title=fields.get("title", current["title"]),
        thumbnail=stored,
        error=page.error,
    )


def _apply_tags(
    conn: sqlite3.Connection, asset_id: int, title: str, host: str
) -> None:
    """Automated tags for a reference: its site, and words from its title.

    Cleared and rewritten, the same as a reprobe does for a file, so a refetch
    of a page that has been retitled does not accumulate both sets. Manual tags
    are untouched by construction - that is what the ``source`` column in the
    primary key is for.
    """
    db.clear_automated_tags(conn, asset_id)

    tags: list[tuple[str, str, float | None]] = []
    if host:
        tags.append((f"source:{vocab.canonical(host)}", "structural", None))
    for token in heuristic.title_tags(title or ""):
        tags.append((token, "heuristic", heuristic.FILENAME_CONFIDENCE))
    db.add_tags(conn, asset_id, tags)


def _store_preview(
    config: Config, page: Page, content_hash: str, replace: bool = False
) -> bool:
    """Fetch the page's own image and keep it as this reference's tile.

    A preview already on disk is left alone unless ``replace``. Refetching an
    unchanged image on every refresh is a download for nothing, and the case
    that genuinely wants a new one - the page redesigned and its ``og:image``
    is different now - is what ``refresh --overwrite`` is for.
    """
    if not page.image_url:
        return False

    existing = thumbs.path_for(config, content_hash)
    if existing.exists():
        if not replace:
            return True
        existing.unlink(missing_ok=True)

    data = reference_probe.fetch_image(page.image_url)
    if data is None:
        return False
    return thumbs.store(config, content_hash, data) is not None
