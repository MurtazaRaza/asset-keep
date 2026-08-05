"""Getting a set of assets back out, with what is known about them attached.

The library's job ends at "these forty are the ones", and the next thing that
happens is always the same: they have to be in a project. Dragging them out one
at a time works and loses everything the index knew - who made them, under what
licence, what they were called before Unity renamed them. So an export is a copy
plus a written record, and the record is the part that is hard to reconstruct
later.

**Nothing is overwritten and nothing is duplicated.** A file already at the
destination with the same content is left alone and reported as unchanged, which
is what makes re-exporting a collection after adding two assets copy two assets.
A file with the same *name* and different content gets ``-1``, because the two
are genuinely different and destroying one of them to avoid a decision is not a
choice this tool gets to make.

**References export as text, not as files.** A link has no bytes; it appears in
the manifest and in ``CREDITS.md`` and nowhere else. That is the honest answer,
and it is why the credits file exists at all - a folder of PNGs cannot carry
attribution, and a sidecar per file is thirty files nobody reads.

The manifest is written for a machine and the credits for a person, and both are
written because they are not the same document. ``assetkeep.json`` re-imports;
``CREDITS.md`` ships.

One property worth knowing rather than designing around: if the destination is
inside a scanned root - which it usually is, since it is usually a Unity
project - the next scan indexes the copies as *additional locations of the same
assets*, because identity is the content hash. The library does not grow, the
tags are already right, and "where is this used" starts answering with both
paths. That falls out of the hash and cost nothing to arrange.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, collection as collection_module, db, hashing
from .config import Config
from .tagging import vocab

log = logging.getLogger(__name__)

MANIFEST_NAME = "assetkeep.json"
CREDITS_NAME = "CREDITS.md"

#: Subfolder per kind, for the layout that uses one. Names chosen to read as
#: project folders rather than as database values: nobody wants a directory
#: called ``model3d``.
KIND_FOLDERS = {
    "image": "Images",
    "model3d": "Models",
    "audio": "Audio",
    "reference": "References",
}

LAYOUTS = ("flat", "kind")


@dataclass
class ExportResult:
    """What one export did, in the terms the CLI and the UI report."""

    destination: Path
    #: Files written, as paths at the destination.
    copied: list[Path] = field(default_factory=list)
    #: Already there with identical content, so not written again.
    unchanged: list[Path] = field(default_factory=list)
    #: Assets with no file to copy: every location gone.
    missing: list[int] = field(default_factory=list)
    #: References, which are recorded rather than copied.
    references: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifest: Path | None = None
    credits: Path | None = None

    @property
    def total(self) -> int:
        return len(self.copied) + len(self.unchanged)

    def as_dict(self) -> dict:
        return {
            "destination": str(self.destination),
            "copied": [str(path) for path in self.copied],
            "unchanged": [str(path) for path in self.unchanged],
            "missing": self.missing,
            "references": self.references,
            "errors": self.errors,
            "manifest": str(self.manifest) if self.manifest else None,
            "credits": str(self.credits) if self.credits else None,
        }


def export_collection(
    conn: sqlite3.Connection,
    config: Config,
    name: str,
    destination: Path,
    folder: str | None = None,
    layout: str = "flat",
    manifest: bool = True,
) -> ExportResult:
    """Export a collection by name, into a subfolder named after it.

    The subfolder is the default rather than an option because the destination
    is nearly always a project's ``Assets/``, and forty loose files dropped into
    the root of one is a mess somebody then has to sort by hand. Pass
    ``folder=""`` to export straight into the destination.
    """
    row = collection_module.by_name(conn, name)
    if row is None:
        raise ValueError(f"no collection {name!r}")

    return export(
        conn,
        config,
        collection_module.members(conn, int(row["id"])),
        destination,
        folder=row["name"] if folder is None else folder,
        layout=layout,
        manifest=manifest,
        title=row["name"],
        notes=row["notes"],
    )


def export(
    conn: sqlite3.Connection,
    config: Config,
    asset_ids,
    destination: Path,
    folder: str | None = None,
    layout: str = "flat",
    manifest: bool = True,
    title: str = "",
    notes: str = "",
) -> ExportResult:
    """Copy assets into ``destination``, with a manifest and a credits file."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; try {' or '.join(LAYOUTS)}")

    destination = Path(destination).expanduser()
    if not destination.is_dir():
        raise ValueError(f"not a directory: {destination}")
    if folder:
        destination = destination / vocab.canonical(folder)
    destination.mkdir(parents=True, exist_ok=True)

    result = ExportResult(destination=destination)
    entries: list[dict] = []

    for asset_id in [int(value) for value in asset_ids]:
        asset = conn.execute(
            "SELECT * FROM asset WHERE id = ?", (asset_id,)
        ).fetchone()
        if asset is None:
            result.errors.append(f"#{asset_id}: no such asset")
            continue

        entry = _describe(conn, asset)

        if asset["kind"] == "reference":
            result.references.append(asset_id)
            entries.append(entry)
            continue

        source = _present_path(conn, asset_id)
        if source is None:
            result.missing.append(asset_id)
            entries.append(entry)
            continue

        try:
            written, fresh = _place(source, destination, layout, asset["kind"])
        except OSError as exc:
            result.errors.append(f"{source.name}: {exc}")
            continue

        (result.copied if fresh else result.unchanged).append(written)
        entry["file"] = written.relative_to(destination).as_posix()
        entries.append(entry)

    if manifest:
        result.manifest = _write_manifest(destination, entries, title, notes)
        result.credits = _write_credits(destination, entries, title)

    return result


def _place(
    source: Path, destination: Path, layout: str, kind: str
) -> tuple[Path, bool]:
    """Copy one file in, returning where it landed and whether it was written.

    Content is compared before anything is written, so an export run twice is
    idempotent rather than a folder full of ``tile-1.png``. The comparison is
    the same hash the index is keyed on, which is what makes "the same file" a
    question with one answer everywhere in this tool.
    """
    target_dir = destination
    if layout == "kind":
        target_dir = destination / KIND_FOLDERS.get(kind, kind)
        target_dir.mkdir(parents=True, exist_ok=True)

    target = target_dir / source.name
    digest = hashing.hash_file(source)

    while target.exists():
        if target.is_file() and hashing.hash_file(target) == digest:
            return target, False
        target = _next_name(target)

    shutil.copy2(source, target)
    return target, True


def _next_name(target: Path) -> Path:
    """``tile.png`` -> ``tile-1.png`` -> ``tile-2.png``.

    >>> _next_name(Path("/x/tile.png")).name
    'tile-1.png'
    >>> _next_name(Path("/x/tile-1.png")).name
    'tile-2.png'
    """
    stem, _, suffix = target.name.rpartition(".")
    stem = stem or target.name
    head, dash, tail = stem.rpartition("-")
    if dash and tail.isdigit():
        stem = f"{head}-{int(tail) + 1}"
    else:
        stem = f"{stem}-1"
    return target.with_name(f"{stem}.{suffix}" if suffix else stem)


def _describe(conn: sqlite3.Connection, asset: sqlite3.Row) -> dict:
    """One asset's row in the manifest: everything a folder of files loses."""
    return {
        "id": int(asset["id"]),
        "kind": asset["kind"],
        "title": asset["title"],
        "hash": asset["content_hash"],
        "source_name": asset["source_name"] or "",
        "source_url": asset["source_url"] or "",
        "license": asset["license"] or "",
        "caption": asset["caption"] or "",
        "notes": asset["notes"] or "",
        "tags": db.tag_names(conn, int(asset["id"])),
    }


def _present_path(conn: sqlite3.Connection, asset_id: int) -> Path | None:
    for row in conn.execute(
        "SELECT abs_path FROM location WHERE asset_id = ? AND present = 1 "
        "ORDER BY id",
        (asset_id,),
    ):
        path = Path(row["abs_path"])
        if path.exists():
            return path
    return None


def _write_manifest(
    destination: Path, entries: list[dict], title: str, notes: str
) -> Path:
    """The machine-readable half: enough to re-import this folder elsewhere."""
    path = destination / MANIFEST_NAME
    path.write_text(
        json.dumps(
            {
                "tool": "assetkeep",
                "version": __version__,
                "exported_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "collection": title,
                "notes": notes,
                "assets": entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


#: Titles listed under one credit heading before the rest are counted instead.
#:
#: Measured on a real export: 126 assets from one Unity project share a single
#: source and licence, and listing all of them makes a 130-line document whose
#: one line of actual attribution is invisible. Twenty is enough to recognise
#: what the group is, and the exhaustive list is in the manifest beside it,
#: which is where a machine would look for it anyway.
CREDIT_ITEMS = 20


def _write_credits(destination: Path, entries: list[dict], title: str) -> Path:
    """The human-readable half: who to credit, and under what licence.

    Grouped by source and licence rather than listed per file, because that is
    the shape the answer takes: forty tiles from one pack are one line of
    attribution, and a per-file list makes a document nobody reads and therefore
    nobody ships.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for entry in entries:
        if entry["kind"] == "reference":
            continue
        key = (entry["source_name"], entry["license"])
        groups.setdefault(key, []).append(entry)

    lines = [f"# Credits{f' - {title}' if title else ''}", ""]
    unknown = groups.pop(("", ""), None)

    for (source, license_name) in sorted(groups):
        items = groups[(source, license_name)]
        heading = source or "Unattributed"
        lines.append(f"## {heading}")
        if license_name:
            lines.append(f"Licence: {license_name}")
        url = next((item["source_url"] for item in items if item["source_url"]), "")
        if url:
            lines.append(url)
        lines.append("")
        lines.extend(_titles(items))
        lines.append("")

    if unknown:
        # Named for what it is. An export whose licensing is unrecorded is a
        # thing to go and fix, not a thing to leave implied by an absence.
        lines.append(f"## No source or licence recorded ({len(unknown)})")
        lines.append("")
        lines.extend(_titles(unknown))
        lines.append("")

    references = [entry for entry in entries if entry["kind"] == "reference"]
    if references:
        lines.append("## References")
        lines.append("")
        for entry in references:
            lines.append(f"- [{entry['title']}]({entry['source_url']})")
        lines.append("")

    path = destination / CREDITS_NAME
    # Explicit UTF-8 rather than the platform default, which on Windows is
    # cp1252 and cannot encode most of what is in this file: an artist called
    # Bögel, a licence line with a © in it, a title pasted from a store page.
    # The failure is a UnicodeEncodeError at the end of an export that has
    # already copied every file.
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _titles(items: list[dict]) -> list[str]:
    """One group's members, capped at :data:`CREDIT_ITEMS`.

    >>> _titles([{"title": "a"}, {"title": "b"}])
    ['- a', '- b']
    >>> _titles([{"title": str(n)} for n in range(25)])[-1]
    '- and 5 more, listed in assetkeep.json'
    """
    lines = [f"- {item['title']}" for item in items[:CREDIT_ITEMS]]
    if len(items) > CREDIT_ITEMS:
        lines.append(
            f"- and {len(items) - CREDIT_ITEMS} more, listed in {MANIFEST_NAME}"
        )
    return lines
