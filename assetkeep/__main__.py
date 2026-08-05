"""Entry point: ``uv run assetkeep``.

The CLI exists for three reasons: it is useful before the UI is written, it is
what scripts and cron talk to, and it is the honest test of whether the index is
actually queryable. Everything the browser will do in M2 goes through the same
functions these commands call.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from . import (
    __version__,
    collection as collection_module,
    config as config_module,
    db,
    desktop,
    export as export_module,
    reference as reference_module,
    scan as scan_module,
    search,
    thumbs,
    vault,
    vectors,
)
from .config import RootConfig
from .probe import audio, model3d
from .tagging import clip, vlm

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    cfg = config_module.load(args.config)
    return args.handler(args, cfg)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assetkeep", description="Local asset index for game development."
    )
    parser.add_argument("--config", type=Path, default=None, help="path to config.toml")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    root = commands.add_parser("root", help="manage scan roots")
    root_commands = root.add_subparsers(dest="root_command", required=True)

    add = root_commands.add_parser("add", help="register a folder to index")
    add.add_argument("path", type=Path)
    add.add_argument(
        "--vendor", action="store_true", help="third-party pack content, not your work"
    )
    add.add_argument(
        "--managed", action="store_true", help="this tool owns the folder's contents"
    )
    add.add_argument("--no-recursive", action="store_true")
    add.add_argument(
        "--exclude", action="append", default=[], metavar="GLOB",
        help="extra exclude pattern, repeatable",
    )
    add.add_argument(
        "--no-default-excludes", action="store_true",
        help="do not merge Library/, Temp/, *.meta and friends",
    )
    add.set_defaults(handler=_root_add)

    listing = root_commands.add_parser("list", help="show configured roots")
    listing.set_defaults(handler=_root_list)

    remove = root_commands.add_parser("remove", help="stop indexing a folder")
    remove.add_argument("path", type=Path)
    remove.set_defaults(handler=_root_remove)

    scan = commands.add_parser("scan", help="bring the index up to date")
    scan.add_argument("roots", nargs="*", type=Path, help="limit to these roots")
    scan.add_argument(
        "--rehash", action="store_true",
        help="hash every file, ignoring the (size, mtime) fast path",
    )
    scan.add_argument(
        "--reprobe", action="store_true",
        help="re-extract attributes and automatic tags for content already "
             "indexed, e.g. after installing assimp. Manual tags are kept.",
    )
    scan.add_argument("--quiet", "-q", action="store_true")
    scan.set_defaults(handler=_scan)

    find = commands.add_parser("search", help="query the index")
    find.add_argument("query", nargs="*", default=[])
    find.add_argument("--limit", type=int, default=50)
    find.add_argument("--offset", type=int, default=0)
    find.add_argument("--paths", action="store_true", help="print paths only")
    find.add_argument("--json", action="store_true")
    find.set_defaults(handler=_search)

    show = commands.add_parser("show", help="everything known about one asset")
    show.add_argument("asset_id", type=int)
    show.add_argument("--json", action="store_true")
    show.set_defaults(handler=_show)

    tag = commands.add_parser("tag", help="add or remove manual tags")
    tag.add_argument("asset_id", type=int)
    tag.add_argument("tags", nargs="+")
    tag.add_argument("--remove", action="store_true")
    tag.set_defaults(handler=_tag)

    edit = commands.add_parser("set", help="edit an asset's metadata")
    edit.add_argument("asset_id", type=int)
    edit.add_argument("--title")
    edit.add_argument("--notes")
    edit.add_argument("--caption")
    edit.add_argument("--license", help="e.g. cc0, cc-by-4.0, proprietary")
    edit.add_argument("--source", dest="source_name", help="who made it")
    edit.add_argument("--source-url", dest="source_url")
    edit.set_defaults(handler=_set)

    collection = commands.add_parser("collection", help="manage collections")
    collection_commands = collection.add_subparsers(
        dest="collection_command", required=True
    )

    collection_list = collection_commands.add_parser("list", help="show collections")
    collection_list.set_defaults(handler=_collection_list)

    collection_new = collection_commands.add_parser("new", help="create a collection")
    collection_new.add_argument("name")
    collection_new.add_argument("--notes", default="")
    collection_new.set_defaults(handler=_collection_new)

    collection_add = collection_commands.add_parser(
        "add", help="add assets, by id or by query"
    )
    collection_add.add_argument("name")
    collection_add.add_argument("assets", nargs="*", type=int, metavar="ID")
    collection_add.add_argument(
        "--query", "-q", help="add everything this search matches"
    )
    collection_add.set_defaults(handler=_collection_add)

    collection_remove = collection_commands.add_parser(
        "remove", help="take assets out of a collection"
    )
    collection_remove.add_argument("name")
    collection_remove.add_argument("assets", nargs="+", type=int, metavar="ID")
    collection_remove.set_defaults(handler=_collection_remove)

    collection_delete = collection_commands.add_parser(
        "delete", help="delete a collection, keeping its assets"
    )
    collection_delete.add_argument("name")
    collection_delete.set_defaults(handler=_collection_delete)

    reference = commands.add_parser(
        "reference", help="assets that are a URL rather than a file"
    )
    reference_commands = reference.add_subparsers(
        dest="reference_command", required=True
    )

    reference_add = reference_commands.add_parser("add", help="index a URL")
    reference_add.add_argument("url")
    reference_add.add_argument("--title", help="overrides the page's own title")
    reference_add.add_argument("--notes", default=None)
    reference_add.add_argument("--license")
    reference_add.add_argument(
        "--tag", action="append", default=[], metavar="TAG", help="repeatable"
    )
    reference_add.add_argument("--collection", help="also add it to this collection")
    reference_add.add_argument(
        "--no-fetch", action="store_true",
        help="do not read the page; take the title from the URL",
    )
    reference_add.set_defaults(handler=_reference_add)

    reference_refresh = reference_commands.add_parser(
        "refresh", help="fetch a reference's page again"
    )
    reference_refresh.add_argument(
        "assets", nargs="*", type=int, metavar="ID", help="default: all of them"
    )
    reference_refresh.add_argument(
        "--overwrite", action="store_true",
        help="replace the title and notes, rather than only filling gaps",
    )
    reference_refresh.set_defaults(handler=_reference_refresh)

    export_command = commands.add_parser(
        "export", help="copy a collection into a project folder"
    )
    export_command.add_argument("name", help="collection name")
    export_command.add_argument(
        "destination", type=Path, nargs="?", default=None,
        help="default: the remembered copy target",
    )
    export_command.add_argument(
        "--layout", choices=export_module.LAYOUTS, default="flat",
        help="flat, or one subfolder per kind",
    )
    export_command.add_argument(
        "--folder", default=None,
        help="subfolder to create; defaults to the collection name, '' for none",
    )
    export_command.add_argument(
        "--no-manifest", action="store_true",
        help=f"skip {export_module.MANIFEST_NAME} and {export_module.CREDITS_NAME}",
    )
    export_command.add_argument(
        "--remember", action="store_true", help="save the destination as copy_target"
    )
    export_command.set_defaults(handler=_export)

    caption = commands.add_parser(
        "caption", help="describe assets with the local vision model"
    )
    caption.add_argument("assets", nargs="*", type=int, metavar="ID")
    caption.add_argument("--query", "-q", help="caption everything this search matches")
    caption.add_argument("--collection", help="caption a collection")
    caption.add_argument(
        "--redo", action="store_true", help="re-describe assets that already have one"
    )
    caption.add_argument("--limit", type=int, default=None)
    caption.set_defaults(handler=_caption)

    vlm_command = commands.add_parser("vlm", help="manage the captioning model")
    vlm_commands = vlm_command.add_subparsers(dest="vlm_command", required=True)

    vlm_status = vlm_commands.add_parser("status", help="what ollama is holding")
    vlm_status.set_defaults(handler=_vlm_status)

    vlm_pull = vlm_commands.add_parser("pull", help="ask ollama to fetch the model")
    vlm_pull.add_argument("--model", default=None, help=f"default {vlm.DEFAULT_MODEL}")
    vlm_pull.set_defaults(handler=_vlm_pull)

    importer = commands.add_parser(
        "import", help="copy files or archives into the vault and index them"
    )
    importer.add_argument("paths", nargs="+", type=Path)
    importer.add_argument(
        "--batch", help="vault subfolder; defaults to the archive or folder name"
    )
    importer.add_argument("--collection", help="also add everything to this collection")
    importer.add_argument(
        "--move", action="store_true", help="remove the source after importing"
    )
    importer.set_defaults(handler=_import)

    prune = commands.add_parser(
        "prune", help="delete assets whose every location is gone"
    )
    prune.add_argument("--dry-run", action="store_true")
    prune.add_argument("--yes", action="store_true", help="skip the confirmation")
    prune.set_defaults(handler=_prune)

    capabilities = commands.add_parser(
        "capabilities", help="which optional extras are live"
    )
    capabilities.set_defaults(handler=_capabilities)

    thumbs_command = commands.add_parser(
        "thumbs", help="generate queued thumbnails in the foreground"
    )
    thumbs_command.add_argument("--limit", type=int, default=None)
    thumbs_command.add_argument(
        "--retry", action="store_true", help="requeue jobs that previously failed"
    )
    thumbs_command.set_defaults(handler=_thumbs)

    model = commands.add_parser("model", help="manage the optional CLIP weights")
    model_commands = model.add_subparsers(dest="model_command", required=True)

    model_status = model_commands.add_parser("status", help="what is installed")
    model_status.set_defaults(handler=_model_status)

    model_download = model_commands.add_parser(
        "download", help="fetch the CLIP weights into ~/AssetKeep/models"
    )
    model_download.add_argument(
        "--model",
        default=None,
        help=f"which variant, default {clip.DEFAULT_VARIANT}",
    )
    model_download.set_defaults(handler=_model_download)

    embed = commands.add_parser(
        "embed", help="compute CLIP embeddings and zero-shot tags"
    )
    embed.add_argument("--limit", type=int, default=None)
    embed.add_argument(
        "--redo",
        action="store_true",
        help="re-embed content that already has a vector",
    )
    embed.set_defaults(handler=_embed)

    serve = commands.add_parser("serve", help="run the browser UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-browser", action="store_true")
    serve.set_defaults(handler=_serve)

    return parser


# --- commands ---------------------------------------------------------------


def _root_add(args, cfg) -> int:
    path = args.path.expanduser().resolve()
    if not path.is_dir():
        print(f"not a directory: {path}", file=sys.stderr)
        return 1

    root = RootConfig(
        path=path,
        mode="managed" if args.managed else "indexed",
        recursive=not args.no_recursive,
        excludes=tuple(args.exclude),
        exclude_defaults=not args.no_default_excludes,
        vendor=args.vendor,
    )
    written = config_module.save(config_module.with_root(cfg, root))
    print(f"added {root.name}  {path}")
    print(f"config: {written}")
    return 0


def _root_list(args, cfg) -> int:
    if not cfg.roots:
        print("no roots configured; try: assetkeep root add <path>")
        return 0

    with db.connect(cfg.db_path) as conn:
        counts = dict(
            conn.execute(
                """
                SELECT r.path, COUNT(DISTINCT l.asset_id) FROM root r
                LEFT JOIN location l ON l.root_id = r.id AND l.present = 1
                GROUP BY r.id
                """
            ).fetchall()
        )

    for root in cfg.roots:
        flags = [root.mode]
        if root.vendor:
            flags.append("vendor")
        if not root.recursive:
            flags.append("flat")
        if not root.enabled:
            flags.append("disabled")
        count = counts.get(str(root.path), 0)
        print(f"{root.name:<24} {count:>7} assets  [{', '.join(flags)}]  {root.path}")
    return 0


def _root_remove(args, cfg) -> int:
    path = args.path.expanduser().resolve()
    if not any(root.path == path for root in cfg.roots):
        print(f"not a configured root: {path}", file=sys.stderr)
        return 1

    config_module.save(config_module.without_root(cfg, path))
    print(f"removed {path}")
    print("indexed assets are kept; run 'assetkeep prune' to drop the missing ones")
    return 0


def _scan(args, cfg) -> int:
    if not cfg.roots:
        print("no roots configured; try: assetkeep root add <path>", file=sys.stderr)
        return 1

    show_progress = not args.quiet and sys.stderr.isatty()

    def progress(stats, path):
        if show_progress and stats.seen % 50 == 0:
            print(f"\r  {stats.seen:>6} files, {stats.hashed} hashed", end="",
                  file=sys.stderr, flush=True)

    conn = db.connect(cfg.db_path)
    try:
        stats = scan_module.scan(
            conn, cfg, only=args.roots or None, rehash=args.rehash,
            reprobe=args.reprobe, progress=progress,
        )
    finally:
        conn.close()

    if show_progress:
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr)

    print(
        f"{stats.roots} root(s), {stats.seen} files: "
        f"{stats.added} new, {stats.relinked} relinked, {stats.updated} changed, "
        f"{stats.unchanged} unchanged, {stats.absent} now missing"
        + (f", {stats.reprobed} reprobed" if stats.reprobed else "")
    )
    for failure in stats.failures[:10]:
        print(f"  ! {failure}", file=sys.stderr)
    if len(stats.failures) > 10:
        print(f"  ! and {len(stats.failures) - 10} more", file=sys.stderr)
    return 0


def _search(args, cfg) -> int:
    query = " ".join(args.query)
    with db.connect(cfg.db_path) as conn:
        rows = search.search(
            conn,
            query,
            limit=args.limit,
            offset=args.offset,
            semantic=clip.ranker(cfg, conn),
        )

        if args.json:
            print(json.dumps([dict(row) for row in rows], indent=2))
            return 0

        for row in rows:
            # A reference's "path" is its URL, which is the thing you would
            # paste somewhere - so --paths stays useful with links in the
            # results rather than printing blank lines.
            path = (
                row["source_url"]
                if row["kind"] == "reference"
                else _primary_path(conn, int(row["id"]))
            )
            if args.paths:
                print(path or "")
                continue
            print(f"{row['id']:>6}  {row['kind']:<8} {_summary(conn, row):<28} {path}")

    if not rows and not args.json:
        print("no matches", file=sys.stderr)
    return 0


def _show(args, cfg) -> int:
    with db.connect(cfg.db_path) as conn:
        asset = conn.execute(
            "SELECT * FROM asset WHERE id = ?", (args.asset_id,)
        ).fetchone()
        if asset is None:
            print(f"no asset {args.asset_id}", file=sys.stderr)
            return 1

        attributes = conn.execute(
            "SELECT key, value_text, value_num FROM attribute WHERE asset_id = ? "
            "ORDER BY key",
            (args.asset_id,),
        ).fetchall()
        tags = conn.execute(
            "SELECT t.name, at.source, at.confidence FROM asset_tag at "
            "JOIN tag t ON t.id = at.tag_id WHERE at.asset_id = ? "
            "ORDER BY at.source, t.name",
            (args.asset_id,),
        ).fetchall()
        locations = conn.execute(
            "SELECT abs_path, size, present FROM location WHERE asset_id = ? "
            "ORDER BY present DESC, abs_path",
            (args.asset_id,),
        ).fetchall()

        if args.json:
            print(json.dumps({
                "asset": dict(asset),
                "attributes": {r["key"]: r["value_text"] if r["value_num"] is None
                               else r["value_num"] for r in attributes},
                "tags": [dict(r) for r in tags],
                "locations": [dict(r) for r in locations],
            }, indent=2))
            return 0

        print(f"#{asset['id']}  {asset['title']}  ({asset['kind']})")
        print(f"  hash     {asset['content_hash']}")
        print(f"  added    {asset['added_at']}")
        for field in ("source_name", "source_url", "license", "caption", "notes"):
            if asset[field]:
                print(f"  {field:<8} {asset[field]}")

        if attributes:
            print("  attributes")
            for row in attributes:
                value = row["value_text"] if row["value_num"] is None else _clean(row["value_num"])
                print(f"    {row['key']:<16} {value}")

        if tags:
            print("  tags")
            for row in tags:
                confidence = f"  {row['confidence']:.2f}" if row["confidence"] else ""
                print(f"    {row['name']:<24} [{row['source']}]{confidence}")

        if locations:
            print("  locations")
            for row in locations:
                marker = " " if row["present"] else "!"
                print(f"    {marker} {row['abs_path']}")
        elif asset["kind"] != "reference":
            print("  locations   none; this asset's every copy has gone")
    return 0


def _tag(args, cfg) -> int:
    with db.connect(cfg.db_path) as conn:
        if conn.execute(
            "SELECT 1 FROM asset WHERE id = ?", (args.asset_id,)
        ).fetchone() is None:
            print(f"no asset {args.asset_id}", file=sys.stderr)
            return 1

        if args.remove:
            for name in args.tags:
                conn.execute(
                    "DELETE FROM asset_tag WHERE asset_id = ? AND tag_id = ? "
                    "AND source = 'manual'",
                    (args.asset_id, db.tag_id(conn, name)),
                )
            action = "removed"
        else:
            db.add_tags(
                conn, args.asset_id, [(name, "manual", None) for name in args.tags]
            )
            action = "added"

        db.touch_asset(conn, args.asset_id)
        db.index_fts(conn, args.asset_id)
        print(f"{action}: {', '.join(db.tag_names(conn, args.asset_id))}")
    return 0


def _set(args, cfg) -> int:
    fields = {
        key: value
        for key, value in vars(args).items()
        if key in db.EDITABLE_FIELDS and value is not None
    }
    if not fields:
        print("nothing to set; try --license or --source", file=sys.stderr)
        return 1

    with db.connect(cfg.db_path) as conn:
        if conn.execute(
            "SELECT 1 FROM asset WHERE id = ?", (args.asset_id,)
        ).fetchone() is None:
            print(f"no asset {args.asset_id}", file=sys.stderr)
            return 1

        db.update_asset(conn, args.asset_id, fields)
        db.index_fts(conn, args.asset_id)

    print(f"#{args.asset_id}: set {', '.join(sorted(fields))}")
    return 0


def _collection_list(args, cfg) -> int:
    with db.connect(cfg.db_path) as conn:
        rows = collection_module.listing(conn)

    if not rows:
        print("no collections; try: assetkeep collection new <name>")
        return 0
    for row in rows:
        notes = f"  {row['notes']}" if row["notes"] else ""
        print(f"{row['name']:<28} {row['count']:>6} assets{notes}")
    return 0


def _collection_new(args, cfg) -> int:
    with db.connect(cfg.db_path) as conn:
        try:
            collection_id = collection_module.create(conn, args.name, args.notes)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        row = collection_module.get(conn, collection_id)

    print(f"{row['name']}  (collection:{row['name']})")
    return 0


def _collection_add(args, cfg) -> int:
    with db.connect(cfg.db_path) as conn:
        ids = list(args.assets)
        if args.query:
            ids += [
                int(row["id"])
                for row in search.search(conn, args.query, limit=100_000)
            ]
        if not ids:
            print("nothing to add; give ids or --query", file=sys.stderr)
            return 1

        collection_id = collection_module.create(conn, args.name)
        added = collection_module.add(conn, collection_id, ids)
        total = len(collection_module.members(conn, collection_id))

    print(f"{args.name}: added {added}, now {total} asset(s)")
    return 0


def _collection_remove(args, cfg) -> int:
    with db.connect(cfg.db_path) as conn:
        row = collection_module.by_name(conn, args.name)
        if row is None:
            print(f"no collection {args.name}", file=sys.stderr)
            return 1
        removed = collection_module.remove(conn, int(row["id"]), args.assets)

    print(f"{row['name']}: removed {removed}")
    return 0


def _collection_delete(args, cfg) -> int:
    with db.connect(cfg.db_path) as conn:
        row = collection_module.by_name(conn, args.name)
        if row is None:
            print(f"no collection {args.name}", file=sys.stderr)
            return 1
        collection_module.delete(conn, int(row["id"]))

    print(f"deleted {row['name']}; its assets are untouched")
    return 0


def _reference_add(args, cfg) -> int:
    with db.connect(cfg.db_path) as conn:
        try:
            result = reference_module.add(
                conn,
                cfg,
                args.url,
                title=args.title,
                notes=args.notes,
                license=args.license,
                tags=args.tag,
                fetch=not args.no_fetch,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        if args.collection:
            collection_id = collection_module.create(conn, args.collection)
            collection_module.add(conn, collection_id, [result.asset_id])

    state = "added" if result.created else "already known, updated"
    print(f"#{result.asset_id}  {result.title}   ({state})")
    print(f"  {result.url}")
    if result.thumbnail:
        print("  preview stored")
    if result.error:
        # Not an error exit: the reference exists and is searchable. The page
        # not answering is a fact about the page.
        print(f"  the page could not be read: {result.error}", file=sys.stderr)
    return 0


def _reference_refresh(args, cfg) -> int:
    with db.connect(cfg.db_path) as conn:
        ids = args.assets or [
            int(row["id"])
            for row in conn.execute(
                "SELECT id FROM asset WHERE kind = 'reference' ORDER BY id"
            )
        ]
        if not ids:
            print("no references to refresh")
            return 0

        failures = 0
        for asset_id in ids:
            result = reference_module.refresh(
                conn, cfg, asset_id, overwrite=args.overwrite
            )
            if result is None:
                print(f"#{asset_id} is not a reference", file=sys.stderr)
                failures += 1
                continue
            marker = "!" if result.error else " "
            print(f"{marker} #{result.asset_id}  {result.title}")
            failures += bool(result.error)

    print(f"{len(ids)} refreshed, {failures} could not be read")
    return 0


def _export(args, cfg) -> int:
    destination = args.destination or cfg.copy_target
    if destination is None:
        print(
            "no destination, and no copy_target configured; "
            "try: assetkeep export <name> <folder> --remember",
            file=sys.stderr,
        )
        return 1

    with db.connect(cfg.db_path) as conn:
        try:
            result = export_module.export_collection(
                conn,
                cfg,
                args.name,
                Path(destination),
                folder=args.folder,
                layout=args.layout,
                manifest=not args.no_manifest,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    print(f"{result.destination}")
    print(
        f"  {len(result.copied)} copied, {len(result.unchanged)} already there, "
        f"{len(result.references)} reference(s) recorded, "
        f"{len(result.missing)} with no file"
    )
    if result.credits:
        print(f"  {result.credits.name} and {result.manifest.name} written")
    for failure in result.errors[:10]:
        print(f"  ! {failure}", file=sys.stderr)

    if args.remember and Path(destination) != cfg.copy_target:
        config_module.save(replace(cfg, copy_target=Path(destination).expanduser()))
        print(f"  remembered {destination} as the copy target")
    return 0


def _caption(args, cfg) -> int:
    """Describe a selection with the vision model, in the foreground."""
    from . import job

    state = vlm.status(cfg)
    if not state["server"]:
        print(f"ollama is not answering at {state['url']}; start it with: ollama serve",
              file=sys.stderr)
        return 1
    if not state["installed"]:
        print(f"{state['model']} is not installed: assetkeep vlm pull",
              file=sys.stderr)
        return 1

    show_progress = sys.stderr.isatty()

    def progress(processed: int, source: Path) -> None:
        if show_progress:
            print(f"\r  {processed:>5}  {source.name[:48]:<48}", end="",
                  file=sys.stderr, flush=True)

    conn = db.connect(cfg.db_path)
    try:
        ids = list(args.assets)
        if args.query:
            ids += [
                int(row["id"])
                for row in search.search(conn, args.query, limit=100_000)
            ]
        if args.collection:
            row = collection_module.by_name(conn, args.collection)
            if row is None:
                print(f"no collection {args.collection}", file=sys.stderr)
                return 1
            ids += collection_module.members(conn, int(row["id"]))
        if not ids:
            print("nothing to caption; give ids, --query or --collection",
                  file=sys.stderr)
            return 1

        queued = job.enqueue_captions(
            conn, ids, redo=args.redo, limit=args.limit
        )
        print(f"{len(queued)} to caption with {state['model']}")
        if not queued:
            return 0

        status = job.drain(conn, cfg, progress=progress)
        written = conn.execute(
            "SELECT COUNT(*) FROM asset WHERE caption IS NOT NULL AND caption != ''"
        ).fetchone()[0]
    finally:
        conn.close()

    if show_progress:
        print("\r" + " " * 60 + "\r", end="", file=sys.stderr)
    print(f"{written} asset(s) captioned in total, {status.failed} failed")
    return 0


def _vlm_status(args, cfg) -> int:
    state = vlm.status(cfg)
    print(f"ollama    {'up' if state['server'] else 'not answering'}   {state['url']}"
          + ("" if state["server"] else "   (ollama serve)"))
    print(f"model     {state['model']}   "
          + ("installed" if state["installed"] else "missing   (assetkeep vlm pull)"))
    if state["models"]:
        print(f"holding   {', '.join(sorted(state['models']))}")

    conn = db.connect(cfg.db_path)
    try:
        captioned = conn.execute(
            "SELECT COUNT(*) FROM asset WHERE caption IS NOT NULL AND caption != ''"
        ).fetchone()[0]
    finally:
        conn.close()
    print(f"captioned {captioned} asset(s)")
    return 0


def _vlm_pull(args, cfg) -> int:
    name = args.model or vlm.model_name(cfg)
    if not vlm.installed_models(cfg):
        print(f"ollama is not answering at {vlm.endpoint(cfg)}", file=sys.stderr)
        return 1

    print(f"pulling {name} through ollama at {vlm.endpoint(cfg)}")
    show_progress = sys.stderr.isatty()

    def progress(status: str, done: int, total: int) -> None:
        if show_progress:
            share = f"{done * 100 // total:>3}%" if total else "    "
            print(f"\r  {status[:40]:<40} {share}", end="", file=sys.stderr, flush=True)

    try:
        vlm.pull(cfg, name, progress=progress)
    except Exception as exc:  # noqa: BLE001 - a failed pull is not a crash
        if show_progress:
            print("", file=sys.stderr)
        print(f"pull failed: {exc}", file=sys.stderr)
        return 1

    if show_progress:
        print("\r" + " " * 52 + "\r", end="", file=sys.stderr)
    print(f"{name} is installed")
    return 0


def _import(args, cfg) -> int:
    conn = db.connect(cfg.db_path)
    try:
        result = vault.import_paths(
            conn, cfg, args.paths, batch=args.batch, move=args.move
        )
        if args.collection and result.imported:
            collection_id = collection_module.create(conn, args.collection)
            collection_module.add(conn, collection_id, result.imported)
    finally:
        conn.close()

    print(
        f"{len(result.imported)} imported, {len(result.duplicates)} already known, "
        f"{len(result.skipped)} skipped"
    )
    for failure in result.errors[:10]:
        print(f"  ! {failure}", file=sys.stderr)
    if result.imported:
        print("run 'assetkeep thumbs' to generate their thumbnails")
    return 0


def _prune(args, cfg) -> int:
    with db.connect(cfg.db_path) as conn:
        doomed = scan_module.prune(conn, dry_run=True)
        if not doomed:
            print("nothing to prune")
            return 0

        print(f"{len(doomed)} asset(s) have no file left:")
        for asset_id, title in doomed[:20]:
            print(f"  #{asset_id}  {title}")
        if len(doomed) > 20:
            print(f"  and {len(doomed) - 20} more")

        if args.dry_run:
            return 0
        if not args.yes:
            # Tags, notes and licence live on the asset and none of it comes
            # back, so the default is to make someone say so.
            reply = input("delete these and everything tagged on them? [y/N] ")
            if reply.strip().lower() not in ("y", "yes"):
                print("cancelled")
                return 1

        scan_module.prune(conn)
        print(f"deleted {len(doomed)} asset(s)")
    return 0


def _thumbs(args, cfg) -> int:
    """Drain the thumbnail queue without starting the server."""
    from . import job

    show_progress = sys.stderr.isatty()

    def progress(processed: int, source: Path) -> None:
        if show_progress:
            print(f"\r  {processed:>6}  {source.name[:48]:<48}", end="",
                  file=sys.stderr, flush=True)

    conn = db.connect(cfg.db_path)
    try:
        if args.retry:
            print(f"requeued {job.requeue_failed(conn)} job(s)")
        status = job.drain(conn, cfg, limit=args.limit, progress=progress)
    finally:
        conn.close()

    if show_progress:
        print("\r" + " " * 60 + "\r", end="", file=sys.stderr)
    print(f"{status.done} done, {status.pending} pending, {status.failed} failed")
    return 0


def _serve(args, cfg) -> int:
    import threading
    import webbrowser

    import uvicorn

    from .server import create_app

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    print(f"\n  AssetKeep {__version__}")
    print(f"  UI:    {url}")
    print(f"  Index: {cfg.db_path}\n")

    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="warning")
    return 0


def _model_status(args, cfg) -> int:
    state = clip.status(cfg)
    print(f"model     {state['model']}   {state['label']}")
    print(f"extra     {'installed' if state['deps'] else 'missing'}"
          + ("" if state["deps"] else "   (uv sync --extra clip)"))
    if state["weights"]:
        print(f"weights   present    {state['path']}")
    else:
        print(f"weights   {_megabytes(state['download_bytes'])} to download"
              "   (assetkeep model download)")

    conn = db.connect(cfg.db_path)
    try:
        embedded = vectors.count(conn, state["model"])
        outstanding = len(vectors.pending(conn, state["model"]))
    finally:
        conn.close()
    print(f"embedded  {embedded} asset(s), {outstanding} outstanding")
    return 0


def _model_download(args, cfg) -> int:
    variant = clip.variant_for(args.model or cfg.tagging.clip_model)
    absent = clip.missing(cfg, variant)

    if not absent:
        print(f"{variant.label} already in {clip.directory(cfg)}")
        return 0

    print(f"downloading {clip.describe_download(variant, absent)}")
    print(f"  into {clip.directory(cfg)}")

    show_progress = sys.stderr.isatty()

    def progress(name: str, done: int, total: int) -> None:
        if show_progress:
            share = done * 100 // max(total, 1)
            print(f"\r  {name[:36]:<36} {share:>3}%", end="", file=sys.stderr,
                  flush=True)

    try:
        written = clip.download(cfg, variant, progress=progress)
    except Exception as exc:  # noqa: BLE001 - a download failure is not a crash
        if show_progress:
            print("", file=sys.stderr)
        print(f"download failed: {exc}", file=sys.stderr)
        return 1

    if show_progress:
        print("\r" + " " * 48 + "\r", end="", file=sys.stderr)
    print(f"{len(written)} file(s) written")
    if not clip.deps_available():
        print("the runtime is still missing: uv sync --extra clip")
    return 0


def _embed(args, cfg) -> int:
    """Fill in whatever the library is missing, in the foreground."""
    from . import job

    if not clip.deps_available():
        print("the clip extra is not installed: uv sync --extra clip",
              file=sys.stderr)
        return 1
    if not clip.installed(cfg):
        print("no weights yet: assetkeep model download", file=sys.stderr)
        return 1

    model = clip.variant_for(cfg.tagging.clip_model).id
    show_progress = sys.stderr.isatty()

    def progress(processed: int, source: Path) -> None:
        if show_progress:
            print(f"\r  {processed:>6}  {source.name[:48]:<48}", end="",
                  file=sys.stderr, flush=True)

    conn = db.connect(cfg.db_path)
    try:
        if args.redo:
            # Everything, rather than only what has no vector. The reason to
            # ask for this is a changed backdrop, preprocessing or model, and
            # none of those show up as a missing row.
            conn.execute("DELETE FROM embedding WHERE model = ?", (model,))
        outstanding = vectors.pending(conn, model, limit=args.limit)
        for asset_id in outstanding:
            db.enqueue(conn, "embedding", asset_id)
        print(f"{len(outstanding)} to embed with {model}")

        status = job.drain(conn, cfg, progress=progress)
        embedded = vectors.count(conn, model)
        # Counted rather than inferred from the queue. An asset with no pixels
        # to look at - an animation-only FBX, an EXR nothing could decode -
        # leaves a finished job and no vector, and saying "708 embedded" out of
        # 762 without saying where the other 54 went is the sort of arithmetic
        # people quite reasonably do not trust.
        nothing = len(vectors.pending(conn, model))
    finally:
        conn.close()

    if show_progress:
        print("\r" + " " * 60 + "\r", end="", file=sys.stderr)
    print(
        f"{embedded} embedded, {status.pending} pending, {status.failed} failed"
        + (f", {nothing} with nothing to look at" if nothing else "")
    )
    return 0


def _capabilities(args, cfg) -> int:
    model = model3d.capabilities(cfg.assimp_lib_path)
    semantic = clip.status(cfg)
    captions = vlm.status(cfg)
    print(f"trimesh   {'yes' if model['trimesh'] else 'no'}   GLTF/GLB/OBJ/STL/PLY")
    print(f"assimp    {model['assimp'] or 'no'}    FBX/DAE/BLEND"
          + ("" if model["assimp"] else _hint("assimp")))
    print(f"ffprobe   {'yes' if audio.available() else 'no'}   OGG/MP3/FLAC/AIFF")
    print(f"ffmpeg    {'yes' if thumbs.available() else 'no'}   "
          "waveforms, levels, EXR"
          + ("" if thumbs.available() else _hint("ffmpeg")))
    print(f"clip      {'yes' if semantic['available'] else 'no'}   "
          f"{semantic['model']}"
          + ("" if semantic["available"] else "   (assetkeep model status)"))
    print(f"vlm       {'yes' if captions['available'] else 'no'}   "
          f"{captions['model']}"
          + ("" if captions["available"] else "   (assetkeep vlm status)"))
    return 0


def _hint(tool: str) -> str:
    """The parenthesised install command beside an absent capability.

    >>> _hint("nothing-installs-this")
    ''
    """
    command = desktop.install_hint(tool)
    return f"   ({command})" if command else ""


def _megabytes(count: int) -> str:
    """
    >>> _megabytes(155845627)
    '149 MB'
    """
    return f"{-(-count // (1024 * 1024))} MB"


# --- output helpers ---------------------------------------------------------


def _primary_path(conn, asset_id: int) -> str | None:
    row = conn.execute(
        "SELECT abs_path FROM location WHERE asset_id = ? "
        "ORDER BY present DESC, id LIMIT 1",
        (asset_id,),
    ).fetchone()
    return row["abs_path"] if row else None


def _summary(conn, asset) -> str:
    """A one-line shape for a result row: dimensions, triangles or duration."""
    if asset["kind"] == "reference":
        row = conn.execute(
            "SELECT value_text FROM attribute WHERE asset_id = ? AND key = 'host'",
            (asset["id"],),
        ).fetchone()
        return row["value_text"] if row else ""

    values = dict(
        conn.execute(
            "SELECT key, value_num FROM attribute WHERE asset_id = ? "
            "AND key IN ('width','height','triangles','duration')",
            (asset["id"],),
        ).fetchall()
    )
    if asset["kind"] == "image" and "width" in values:
        return f"{_clean(values['width'])}x{_clean(values['height'])}"
    if asset["kind"] == "model3d" and "triangles" in values:
        return f"{_clean(values['triangles'])} tris"
    if asset["kind"] == "audio" and "duration" in values:
        return f"{values['duration']:.1f}s"
    return ""


def _clean(value: float) -> str:
    """Numbers come back from SQLite as floats; sizes are not 512.0 pixels."""
    return str(int(value)) if float(value).is_integer() else str(value)


if __name__ == "__main__":
    sys.exit(main())
