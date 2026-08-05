# AssetKeep - Plan

A local asset index for game development. Point it at folders, it extracts what it can,
tags what it can, and makes everything findable later. No server, no subscription, no
cloud.

## Scope for v1

In scope: images (sprites, textures, tilesets, UI), 3D models and animations, audio
(structural metadata only), and references (URLs to asset stores, tutorials, reference
art).

**Out of scope: Unity-native files.** Prefabs, materials, scenes and animator controllers
are over half a typical project by file count, but they are Unity-specific composites
rather than portable assets, and indexing them needs a YAML parser plus GUID resolution
through `.meta` files. Excluded deliberately, not overlooked.

Deliberately deferred: any Obsidian integration.

## Calibration

Defaults are tuned against a real project rather than guessed:
`_UnityProjects/MyStarterUtils`, 5,323 files in `Assets/`.

| Measurement | Value | What it decides |
|---|---|---|
| `Library/` vs `Assets/` | 3.4 GB vs 576 MB | Excludes are not optional |
| `.meta` vs real files | 6,094 vs 5,323 | Excluding `.meta` roughly halves the index |
| FBX vs OBJ | 223 vs 4 | FBX support is required, not an extra |
| Images already under 256px | 36% | Skip-small avoids a third of all thumbnails |
| Images with 256 colours or fewer | 65% | Lossless WebP is the common path |
| Images with alpha | 53% | Checkerboard compositing is the common case |
| Median image size | 14 KB | Thumbnails for the whole project cost under 3 MB |

Every image in that project is third-party pack content. That is the normal case, and it
is why roots carry a `vendor` flag: separating "assets I can use" from "work I made" is a
distinction the library needs from the start.

## Principles

**Identity is the content hash, not the path.** Files in game development move
constantly. Reorganising `Assets/`, re-exporting, copying a pack into a second project.
If identity is the path, the library rots within a month. With a content hash, a moved
file is silently re-linked on the next scan and duplicates collapse on their own. One
asset can have many locations.

**Three layers, each independently rebuildable.**

1. The files, wherever they live. Untouched.
2. The metadata: durable, portable, human-editable.
3. The index (SQLite): purely derived, deletable at any time, rebuilt by a rescan.

Nothing exists only in layer 3. That is what makes the eventual sync and Obsidian
conversations easy: you never sync a database, you sync layer 2, which is small text.

**Tag provenance is recorded per tag.** Every tag knows whether it came from the file
itself, a heuristic, CLIP, a VLM, or from you. Auto tags are regenerable and get replaced
on rescan. Manual tags are never touched by any automated pass. Without this you can
never safely re-run the tagger after improving it, and you will want to improve it.

**Degrade gracefully.** Optional capabilities (CLIP, FBX parsing, VLM captions) are
detected at runtime. Missing ones hide their controls rather than erroring. Base install
is numpy, PIL and stdlib.

## Custody: hybrid

Two modes, chosen per root:

- **Indexed in place.** The default. Existing folders work untouched: the
  `AssetGeneratorHelper/out` batches, Unity project `Assets/` trees, extracted packs.
  The tool never writes a byte into these.
- **Managed vault.** An opt-in content-addressed store at `~/AssetKeep/vault/` for loose
  downloads that have no home yet. Dragging a zip from itch.io into the UI imports,
  extracts and files it.

The hash means an asset can be both: imported into the vault and later copied into a
Unity project, still recognised as one asset in two locations.

## Scan roots

A managed list of folders in config, each searched recursively by default. Added via
`assetkeep root add <path>` or a folder picker in the UI.

```toml
[[root]]
path = "~/UnityProjects/Prototype/Assets"
mode = "indexed"          # or "managed"
recursive = true
exclude = ["**/Library/**", "**/Temp/**", "*.meta"]
```

Two defaults matter more than they sound:

- **Aggressive default excludes.** `Library/`, `Temp/`, `Obj/`, `Build/`, `Logs/`,
  `.git/`, `node_modules/`, `.DS_Store`. A single Unity project's `Library/` folder can
  hold six figures of cached intermediates and will drown the index if left in.
- **Symlinks are not followed.** Asset folders cross-link constantly and cycles are easy.

Unity `.meta` files are excluded as assets but worth parsing later: they carry the GUID
Unity uses to track a file across renames, which pairs naturally with hash identity.

## Data model

```
asset          id, kind, content_hash, title, notes,
               source_url, source_name, license, managed,
               added_at, updated_at
location       asset_id, abs_path, size, mtime, last_seen, present
tag            id, name (canonical kebab), namespace
asset_tag      asset_id, tag_id, source, confidence, created_at
attribute      asset_id, key, value_text, value_num      (kind-specific structural data)
embedding      asset_id, model, vector                    (sqlite-vec)
collection     id, name, notes
collection_asset
```

`asset_fts` (FTS5) indexes title, notes, filename, caption and tag names.

### Source and licence

Deliberately first-class fields rather than tags: where an asset came from, who made it,
what licence, whether it is commercially usable. Free to capture at import, effectively
unreconstructable a year later when you are trying to ship. Search can filter on it
(`license:cc0`).

## Ingest pipeline

```
scan(root)
  walk
    (size, mtime) unchanged since last_seen?  -> touch location, skip
    hash
      hash known?  -> add/update location, done
      new          -> probe -> extract -> tag -> enqueue thumbnail + embedding
```

Hashing only on `(size, mtime)` change is what keeps a rescan of a 50k-file Unity project
fast. Optional `watchdog` watcher for live folders.

## Auto-tagging, in tiers

**Tier 1, structural.** Always on, no models. Type, dimensions, alpha, colour count,
dominant palette, aspect, file size, dates. For 3D: triangle count, bounds, material
count, whether it has a rig or animation clips.

**Tier 2, heuristic.** Always on, no models. Filename tokenisation (`goblin_walk_01.png`
gives `goblin`, `walk`), parent folder names as tags, spritesheet grid detection,
tileability by edge-wrap check, normal-map detection by blue dominance, pixel-art block
size detection. That last one already exists inside AssetGeneratorHelper's
post-processing and can be lifted.

A sibling image sharing a model's basename is used as that model's thumbnail. Packs ship
these constantly and it beats any render.

**Tier 3, CLIP.** Optional extra, ~150 MB, comfortable on an M1. One model, three jobs:

- semantic text search over the library
- visual similarity, "more like this"
- zero-shot tagging: score each image against a curated game-dev vocabulary
  (`tileset`, `character-sprite`, `ui-icon`, `weapon`, `prop`, `portrait`, `vfx`,
  `normal-map`, `font-atlas`, ...) and keep what clears a threshold

Scoring against a fixed vocabulary rather than generating prose is the important choice.
It yields canonical, filterable tags, and a small model does it far better than it writes
captions.

**Tier 4, text LLM.** Uses the already-installed `qwen2.5:3b` via ollama. Canonicalising
tag aliases, tidying filename tokens, turning a fetched page title into tags for
reference assets.

**Tier 5, VLM captions.** Opt-in, batched, background queue. `moondream` (~1.7 GB) is the
realistic ceiling alongside everything else on 8 GB. Applied per-asset or per-collection
on request, not to the whole library.

## 3D handling

FBX outnumbers OBJ 56 to 1 in real projects here, so FBX is the primary path rather than
a degraded one. It is read through `assimp`, installed with `brew install assimp` and
bound via `impasse`, a maintained cffi binding. `pyassimp` is the fallback but is prone
to failing to locate the library.

`trimesh` remains the pure-Python path for GLTF, GLB, OBJ, STL and PLY, so those work
with no system dependency at all. Without assimp the tool still runs, and FBX degrades to
filename and file-level metadata with `/api/capabilities` reporting the gap.

Thumbnails come from a small numpy z-buffer rasteriser (flat-shaded triangles, ~50 lines)
rather than an OpenGL stack, because headless GL on macOS is a fight with no payoff at
256px. Blender, if ever installed, slots in as an optional better backend.

**A sibling image sharing the model's basename wins over any render.** Asset packs ship
these constantly and a supplied preview beats a flat-shaded one.

## Audio

Structural metadata, no fingerprinting and no BPM detection: duration, sample rate,
channels, bit depth and codec. WAV parses through the stdlib `wave` module. OGG and
anything else goes through the already-installed `ffprobe`.

One number is measured rather than parsed: **loudness**, as peak and RMS in dBFS. It
costs a decode, and it earns it by answering the question no header field can - why one
sound effect is so much quieter than the rest of the pack, and which files are clipping.

Thumbnails are rendered waveform peaks, which makes an audio file recognisable in a grid
built for images. They are drawn at absolute scale rather than peak-normalised, so that
a wall of them compares files rather than showing each one in isolation, with amplitude
square-rooted so a quiet file is still a shape and not an empty box. Columns that reach
full scale are drawn in a warning colour.

Playback is a mode rather than a modal: `p` auditions the sound under the cursor and
keeps following the cursor as it moves, because browsing sound effects means hearing a
hundred of them in a row. Quick Look stretches the waveform wide, puts a playhead on it,
and seeks where you click.

## Thumbnails

One size, 256px longest edge, WebP, stored at `~/AssetKeep/thumbs/ab/cd/<hash>.webp`.
Sharded by hash prefix so no directory holds ten thousand entries, content-addressed so
duplicates share one file, and never inside SQLite, which has to stay disposable.

**If the source is already smaller than the thumb box, no thumbnail is generated at all
and the original is served.** A 32x32 sprite is around 1 KB and any thumbnail of it would
be larger. Most of a pixel-art library therefore never generates a thumbnail file.

Encoding is chosen by content: 256 colours or fewer goes lossless WebP, which is tiny on
flat art, and everything else goes lossy WebP at q80. Roughly 2 to 6 KB for sprite-like
art, 8 to 20 KB for photographic textures. A 10,000 asset library lands well under
100 MB, and with the skip-small rule, realistically half that.

Two rendering rules, both in the UI rather than baked into the file: scale with
nearest-neighbour so pixel art stays crisp, and composite transparency over a CSS
checkerboard.

## Similarity

Perceptual hashing (dHash) and dominant-palette distance live in the core probe, not the
optional tier. Both are pure numpy, cost nothing at scan time, and give "find visually
similar" and near-duplicate detection from day one with no model installed. CLIP later
upgrades this from visual similarity to semantic similarity without replacing it.

## Search

One query bar over a small grammar, compiled to SQL plus an optional vector pass:

```
dark cave  kind:image  tag:tileset  -tag:wip  license:cc0  w:>=512  has:alpha
```

Bare words hit FTS. With CLIP present, they also run as a semantic pass and results merge
by score.

## Interface

FastAPI + uvicorn backend, vanilla JS frontend with no build step, matching
AssetGeneratorHelper. A CLI covers `scan`, `search`, `tag`, `show`, `export`, for
scripting and for use before the UI exists.

### Layout: two-pane with a slide-over inspector

The tool has two jobs that pull in opposite directions: browsing, where density wins, and
tagging, where the metadata editor wants width. On a 13" M1 Air at an effective 1440px, a
permanently docked three-pane layout spends around 480px on chrome and leaves roughly
700px of grid, which is four or five thumbnails per row.

So: a collapsible ~200px sidebar, the grid taking everything else, and the inspector
sliding in from the right over the grid rather than shrinking it. Escape closes it. That
buys roughly 40 percent more visible thumbnails while still giving the editor real width
on demand.

**Sidebar**: roots as a tree, tags sorted by frequency and filterable, collections, saved
searches.

**Main**: search bar, a row of active filter chips, then the grid.

### The grid

Virtualised, rendering only the visible window plus a buffer, so a 50,000 item library
scrolls at full speed. Fixed square cells rather than a justified masonry layout: with
mixed aspect ratios, uniform cells scan far better and sprites are what matters here.
Images fit with `object-fit: contain`, nearest-neighbour scaling, checkerboard behind
transparency. Cell size steps through 96 / 128 / 192 / 256px.

Selection is click, shift-click for a range, cmd-click to toggle. Hovering reveals
filename, dimensions and a kind badge. A status bar carries result count, selection
count, scan progress and the background job queue.

### Filter chips are the single source of truth

Clicking a tag in the sidebar does not maintain separate state from the search bar. It
appends `tag:x` to the query. Every active filter, wherever it came from, renders as one
removable chip. The query bar and the sidebar are two views of the same thing, which
avoids the usual mess where clicking a facet and typing a query disagree about what is
being shown.

### Quick Look

Space opens a fullscreen preview. Arrow keys continue moving through results while it is
open, space closes it. Detected spritesheets play as an animation, and 3D models scrub
through a pre-rendered turntable strip rather than shipping a WebGL viewer.

### Keyboard map

Browsing-first, so navigation is reachable without the mouse:

```
/       focus search          s       find similar to selection
arrows  move in grid          r       reveal in Finder
space   quick look            c       copy to destination
enter   open inspector        C       copy path to clipboard
esc     close, then deselect  t       focus tag input
[       toggle sidebar        +/-     thumbnail size
```

### Getting assets out

All four paths, since each covers a different moment:

- **Reveal in Finder**, always works.
- **Copy to destination**, with a remembered target folder (the Unity `Assets/` path you
  are currently working in). The fastest common case.
- **Copy path to clipboard**, for terminals, scripts and file dialogs.
- **Drag out of the browser**, as progressive enhancement via the `DownloadURL` drag
  data. This is real in Chromium and absent elsewhere, so it is built as a bonus on top
  of the other three rather than as the primary path.

## Layout

```
assetkeep/
  __main__.py       CLI
  config.py         TOML config: roots, vault path, vocabulary
  db.py             schema + migrations
  hashing.py
  scan.py           walker, incremental logic
  roots.py          root registry, exclude matching, symlink policy
  probe/
    image.py        PIL: dims, alpha, palette, spritesheet, tileability, pixel block
    model3d.py      assimp for FBX, trimesh for GLTF/OBJ/STL/PLY
    audio.py        stdlib wave, ffprobe for everything else
    reference.py    URL metadata fetch (title, og:image, favicon)
  similarity.py     dHash + palette distance (core, no models)
  tagging/
    heuristic.py    filename and folder tokenisation
    vocab.py        curated game-dev vocabulary + aliases
    clip.py         embeddings + zero-shot vocab tagging   [extra: clip]
    llm.py          ollama text normalisation              [extra: llm]
    vlm.py          ollama captioning                      [extra: vlm]
  thumbs.py         PIL for images, numpy rasteriser for 3D
  search.py         query grammar -> SQL + vector
  server.py         FastAPI
  web/              index.html, app.js, style.css
```

Optional extras in `pyproject.toml`: `clip`, `model3d`, `vlm`, mirroring the existing
`segment` extra pattern.

## Milestones

Ordered for browsing first, since that is the activity the UI is tuned for.

**M1 - Core index.** Config, root registry, schema, hashing, incremental scan, image, 3D
and audio structural probes, perceptual hash and palette, heuristic tagging, CLI `scan` /
`search` / `tag` / `show`. No models, no UI. Useful on its own the day it lands.

**M2 - Browsing UI.** Thumbnail generation, virtualised grid, sidebar facets, query bar
with filter chips, Quick Look, find-similar on dHash, and all four retrieval actions.
This is the milestone that makes the tool feel real.

**M3 - Curation.** Slide-over inspector, manual tagging with autocomplete, bulk tag
editing across a selection, collections, vault import by drag and drop.

**M4 - Semantic discovery.** CLIP embeddings, semantic search, zero-shot vocabulary
tagging, similarity upgraded from visual to semantic.

**M5 - References and export.** URL assets with metadata fetch, export bundles, copy a
collection into a Unity project, optional VLM captions.

**M6 - Audio, properly.** Audio was indexed from M1 and never finished: the fields were
probed and shown nowhere, the waveform was drawn in a way that threw the interesting
information away, and the only way to hear anything was a fullscreen modal. So: loudness
measured and made searchable, waveforms that show how loud a file is and mark the ones
that clip, `rate:` / `channels:` / `depth:` / `peak:` filters, audition from the grid, a
wide scrubbable waveform in Quick Look, and the first tests audio has ever had.

**Later.** Obsidian bridge, sync.

## Deferred, with reasons

- **Obsidian.** The intended shape is per-asset markdown with frontmatter written into a
  vault subfolder, so native linking and backlinks work and metadata sync comes free from
  the existing vault sync. Deferred until the core tool is in daily use and it is clear
  what is actually worth linking. Keeping metadata in a clean layer 2 keeps this open.
- **Sync.** Falls out of the layer split: sync layer 2 as text, decide separately about
  binaries.
- **Audio.** Out of v1 scope by choice.
