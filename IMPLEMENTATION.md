# AssetKeep - Implementation

Buildable spec. Design rationale lives in [PLAN.md](PLAN.md); this document is the
contract for what actually gets written.

## Stack

Python 3.12, managed with `uv`. FastAPI + uvicorn backend, vanilla ES-module frontend
with no build step, matching AssetGeneratorHelper.

```toml
[project]
name = "assetkeep"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pillow>=10.4",
    "numpy>=2.0",
    "httpx>=0.27",
    "tomli-w>=1.0",
]

[project.optional-dependencies]
model3d = ["trimesh>=4.4", "impasse>=1.0"]
clip    = ["onnxruntime>=1.19", "tokenizers>=0.20"]
watch   = ["watchdog>=5.0"]

[project.scripts]
assetkeep = "assetkeep.__main__:main"

[dependency-groups]
dev = ["pytest>=8.3", "pytest-asyncio>=0.24"]
```

**CLIP runs through onnxruntime, not torch.** The exported ViT-B/32 towers are 580 MB
against about 2 GB for a torch install, which matters on an 8 GB machine that also
runs ComfyUI. Nothing in this tier needs autograd, a training loop or a GPU. The VLM tier
needs no dependency at all, since it talks to the existing ollama HTTP API through
`httpx`.

Base install is Pillow, numpy and stdlib. Everything else is detected at runtime.

### System dependencies

Both optional, both detected at import and reported through `/api/capabilities`.

```bash
brew install assimp      # FBX parsing. Already-present ffmpeg covers the rest.
```

- **assimp** backs FBX, which is 98 percent of real 3D content here. Bound through
  `impasse`, a maintained cffi binding, with `pyassimp` as fallback. `pyassimp` is not
  the primary because it is poorly maintained and routinely fails to locate the native
  library. The loader tries `impasse`, then `pyassimp`, then an explicit
  `assimp_lib_path` from config, and only then gives up.
- **ffmpeg / ffprobe**, already installed at `/opt/homebrew/bin`, covers OGG and any
  non-WAV audio, and decodes EXR for thumbnails since Pillow cannot.

## Paths

```
~/.config/assetkeep/config.toml     configuration
~/AssetKeep/index.db                SQLite index (disposable, rebuildable)
~/AssetKeep/thumbs/ab/cd/<hash>.webp
~/AssetKeep/vault/                  managed content-addressed store
~/AssetKeep/models/                 downloaded CLIP weights
```

## Configuration

```toml
[general]
db_path        = "~/AssetKeep/index.db"
vault_path     = "~/AssetKeep/vault"
thumbs_path    = "~/AssetKeep/thumbs"
copy_target    = "~/UnityProjects/Prototype/Assets"   # remembered "copy to" destination
missing_policy = "keep"                               # keep | delete

[thumbnails]
max_edge     = 256
quality      = 80
skip_smaller = true    # do not generate when the source already fits the box

[[root]]
path      = "~/_UnityProjects/MyStarterUtils/MyStarterUtils/Assets"
mode      = "indexed"       # indexed | managed
recursive = true
vendor    = true            # third-party pack content, not my own work
exclude   = ["**/Library/**", "**/Temp/**"]

[[root]]
path   = "~/_AutomationProjs/AssetGeneratorHelper/out"
vendor = false

[tagging]
vocab_path      = ""      # optional user vocabulary override
clip_model      = "clip-vit-b-32"   # or clip-vit-b-32-int8, a quarter the size
clip_threshold  = 0.4     # a probability within a namespace, not a cosine
folder_depth    = 2       # how many parent folder names become tags

[vlm]
model   = "moondream"                # captioning model, held by ollama
url     = "http://127.0.0.1:11434"   # ollama need not be on this machine
timeout = 180.0                      # the first call loads 1.7 GB off disk
```

Default excludes are merged into every root unless it sets `exclude_defaults = false`:

```
**/Library/**  **/Temp/**  **/Obj/**  **/Build/**  **/Logs/**
**/.git/**     **/node_modules/**     .DS_Store    *.meta
```

`vendor` marks a root as third-party pack content. Since realistically most indexed
imagery is purchased or downloaded packs, `is:vendor` and `-is:vendor` are how you
separate what you can use from what you made.

### Known extensions

An allowlist, not a blocklist. Anything unlisted is ignored, which is what keeps the
1,243 `.cs` files and the roughly 2,800 Unity-native YAML files out without needing a
single exclude rule.

```python
IMAGE  = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tga", ".tif", ".tiff",
          ".psd", ".exr", ".aseprite", ".ase"}
MODEL  = {".fbx", ".gltf", ".glb", ".obj", ".stl", ".ply", ".dae", ".blend"}
AUDIO  = {".wav", ".ogg", ".mp3", ".flac", ".aiff", ".aif"}
```

`.psd` reads as a flattened composite through Pillow. `.exr` gets dimensions from a
direct header parse and its thumbnail through ffmpeg. `.aseprite` gets header-parsed
dimensions and frame count, with no thumbnail unless the file embeds a preview.

## Schema

`PRAGMA user_version` carries the schema version; migrations are ordered functions in
`db.py`. `PRAGMA journal_mode = WAL` and `foreign_keys = ON` on every connection.

```sql
CREATE TABLE root (
    id        INTEGER PRIMARY KEY,
    path      TEXT NOT NULL UNIQUE,
    mode      TEXT NOT NULL DEFAULT 'indexed',
    recursive INTEGER NOT NULL DEFAULT 1,
    excludes  TEXT NOT NULL DEFAULT '[]',
    vendor    INTEGER NOT NULL DEFAULT 0,
    enabled   INTEGER NOT NULL DEFAULT 1,
    last_scan TEXT
);

CREATE TABLE asset (
    id           INTEGER PRIMARY KEY,
    kind         TEXT NOT NULL,             -- image | model3d | audio | reference
    content_hash TEXT UNIQUE,               -- of the bytes; of the URL for a reference
    title        TEXT NOT NULL,
    notes        TEXT NOT NULL DEFAULT '',
    caption      TEXT,
    source_url   TEXT,
    source_name  TEXT,
    license      TEXT,
    managed      INTEGER NOT NULL DEFAULT 0,
    added_at     TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE location (
    id        INTEGER PRIMARY KEY,
    asset_id  INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
    root_id   INTEGER REFERENCES root(id) ON DELETE SET NULL,
    abs_path  TEXT NOT NULL UNIQUE,
    size      INTEGER NOT NULL,
    mtime     REAL NOT NULL,
    last_seen TEXT NOT NULL,
    present   INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX location_asset ON location(asset_id);

CREATE TABLE tag (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE,         -- canonical kebab-case
    namespace TEXT                          -- type | style | subject | source
);

CREATE TABLE tag_alias (
    alias  TEXT PRIMARY KEY,
    tag_id INTEGER NOT NULL REFERENCES tag(id) ON DELETE CASCADE
);

CREATE TABLE asset_tag (
    asset_id   INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
    tag_id     INTEGER NOT NULL REFERENCES tag(id)   ON DELETE CASCADE,
    source     TEXT NOT NULL,               -- manual | structural | heuristic | clip | vlm | imported
    confidence REAL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (asset_id, tag_id, source)
);
CREATE INDEX asset_tag_tag ON asset_tag(tag_id);

CREATE TABLE attribute (
    asset_id   INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value_text TEXT,
    value_num  REAL,
    PRIMARY KEY (asset_id, key)
);
CREATE INDEX attribute_num ON attribute(key, value_num);

CREATE TABLE phash (
    asset_id INTEGER PRIMARY KEY REFERENCES asset(id) ON DELETE CASCADE,
    dhash    INTEGER NOT NULL,              -- 64-bit
    palette  BLOB NOT NULL                  -- 8 packed RGB triples, frequency ordered
);

CREATE TABLE embedding (
    asset_id INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
    model    TEXT NOT NULL,
    vector   BLOB NOT NULL,                 -- float32
    PRIMARY KEY (asset_id, model)
);

CREATE TABLE collection (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    notes      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE collection_asset (
    collection_id INTEGER NOT NULL REFERENCES collection(id) ON DELETE CASCADE,
    asset_id      INTEGER NOT NULL REFERENCES asset(id)      ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    PRIMARY KEY (collection_id, asset_id)
);

CREATE TABLE job (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL,               -- thumbnail | embedding | caption
    asset_id   INTEGER REFERENCES asset(id) ON DELETE CASCADE,
    state      TEXT NOT NULL DEFAULT 'pending',
    attempts   INTEGER NOT NULL DEFAULT 0,
    error      TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX job_pending ON job(state, kind);

CREATE VIRTUAL TABLE asset_fts USING fts5(
    title, notes, filename, caption, tags,
    tokenize = 'unicode61 remove_diacritics 2'
);
```

### The one schema decision that carries weight

`asset_tag`'s primary key is `(asset_id, tag_id, source)`, not `(asset_id, tag_id)`.

The same tag can therefore exist twice on one asset, once as `heuristic` and once as
`manual`. A rescan deletes every row for an asset whose source is in the automated set
and reinserts fresh ones, and manual rows are untouched by construction rather than by a
careful `WHERE` clause someone will eventually get wrong. Regenerating tags after
improving the tagger is then a safe, boring operation, which it has to be, because it
will happen often.

## Scan pipeline

```python
for root in enabled_roots():
    for path in walk(root.path, recursive=root.recursive, follow_symlinks=False):
        if matches_excludes(path, root): continue
        if path.suffix.lower() not in KNOWN_EXTENSIONS: continue

        st  = path.stat()
        loc = location_by_path(path)

        # Fast path: unchanged since last scan, never hashed.
        if loc and loc.size == st.st_size and loc.mtime == st.st_mtime:
            touch_last_seen(loc); continue

        digest = blake2b_file(path)
        asset  = asset_by_hash(digest)

        if asset is None:
            asset = create_asset(kind_for(path), digest, path)
            attrs, tags = probe(path, asset.kind)
            write_attributes(asset, attrs)
            write_auto_tags(asset, tags)
            enqueue(asset, "thumbnail")
            if clip_available(): enqueue(asset, "embedding")

        upsert_location(asset, path, st, root)

    mark_unseen_absent(root)
```

`mark_unseen_absent` sets `present = 0`. It does **not** delete. An unplugged external
drive or a temporarily moved folder must not destroy tags that took real effort to apply.
Assets with no present locations surface under `is:missing`, and pruning is an explicit
`assetkeep prune` command.

Hashing is blake2b at 64 KB chunks. The `(size, mtime)` fast path is what makes a rescan
of a large Unity tree take seconds instead of minutes.

## Probes

Each probe returns `(attributes: dict, tags: list[tuple[name, source, confidence]])`.

### image.py

Attributes: `width`, `height`, `aspect`, `mode`, `has_alpha`, `color_count`, `bytes`,
`frame_count`.

Heuristics, all numpy and PIL:

| Detection | Method | Tag |
|---|---|---|
| Pixel art | modal run length of colour blocks along both axes; a consistent block size > 1 means the image is an upscaled pixel grid | `pixel-art`, attr `block_size` |
| Spritesheet | uniform grid of fully transparent gutters, or a divisor of both dimensions with repeating alpha structure | `spritesheet`, attrs `cols`, `rows` |
| Tileable | mean absolute difference between opposite edge columns and rows below a threshold | `tileable` |
| Normal map | blue channel dominant, mean near (128, 128, 255) | `normal-map` |
| Mask | single channel or fully desaturated with two dominant values | `mask` |
| Transparent | alpha channel with any pixel below 255 | `has-alpha` |

The pixel-art block-size detection already exists in AssetGeneratorHelper's
post-processing and should be lifted rather than rewritten.

### model3d.py

Attributes: `triangles`, `vertices`, `materials`, `bounds_x/y/z`, `has_rig`,
`bone_count`, `animation_count`.

Two backends behind one interface, chosen by extension:

- **assimp** for FBX, which is 223 of the 227 model files measured. Loader order is
  `impasse`, then `pyassimp`, then an explicit `assimp_lib_path` from config.
- **trimesh** for GLTF, GLB, OBJ, STL and PLY, pure Python, no system dependency.

If assimp is missing the tool still runs and FBX degrades to file-level metadata, with
the gap reported through `/api/capabilities` so the UI can say so rather than silently
showing 223 blank tiles.

Thumbnails come from a numpy z-buffer rasteriser in `thumbs.py`: flat-shaded triangles,
single directional light, fitted to bounds. Roughly 50 lines, no GL context, works
headless. Blender, if ever installed, slots in behind the same interface.

**Before rendering anything, check for a sibling image sharing the model's basename.**
Asset packs ship these constantly and a supplied preview beats any render.

### audio.py

No fingerprinting, no BPM detection.

Attributes: `duration`, `sample_rate`, `channels`, `bit_depth`, `bitrate`, `codec`,
`peak_db`, `rms_db`. WAV parses through the stdlib `wave` module with no subprocess -
3.1 ms against ffprobe's 26 ms, on 72% of a game project's audio. Everything else goes
through `ffprobe -v quiet -print_format json -show_streams`.

Bit depth comes from `bits_per_sample`, falling back to `bits_per_raw_sample`. That
order is the whole of the fix: ffprobe leaves the second unset for 16-bit PCM, and
reports `0` in the first for lossy codecs, where the concept does not apply and
`bitrate` is what to record instead.

Loudness is the one field that is measured rather than read: `decode_pcm` pulls mono
16-bit samples at 8 kHz through ffmpeg, and peak and RMS are taken over them in dBFS
with a -96 dB floor. It costs 36 ms a file. `is:clipping` is peak at or above -0.1 dB
and `is:silent` is peak below -40 dB, and both thresholds live here so that the
waveform tile and the query bar cannot drift apart - which they did, and which no test
caught, because each half was self-consistent.

Heuristic tags from duration, since it is the one dimension that reliably separates
categories in a game project: under 2 seconds tags `sfx`, over 30 seconds tags `music`,
and the range between is left untagged rather than guessed at.

Thumbnails are rendered waveform peaks: `min`/`max` per column via `reduceat` over
computed edges, drawn at absolute scale with square-rooted amplitude so that loudness
survives into the grid, and clipped columns in a warning colour. `reduceat` rather than
a reshape because the reshape truncated the tail to make the length divide evenly, which
silently dropped a fifth of a 300-sample clip and drew anything under 256 samples as a
flat line. Without any of this an audio file is an invisible row in a grid built for
images.

Playback is `web/audition.js` for the grid and an `<audio>` element under a wide
scrubbable waveform in Quick Look; the two stop each other, since there is one pair of
ears.

### reference.py

Fetches the URL, extracts `<title>`, `og:description` and `og:image`. The og:image
becomes the thumbnail. No archiving in v1.

Split in two when it was built, along the line every other probe already sits on:
`probe/reference.py` fetches and parses and touches no database, and `reference.py`
at the top level is the add-URL flow that decides what to record. Only the first
half needs the network, which is what makes the second half testable.

A reference's `content_hash` is `blake2b` of its normalised URL rather than `NULL`.
Three things fall out of that and none fall out of `NULL`: adding the same link
twice is one asset, because the column is already `UNIQUE`; the `og:image` lands in
the content-addressed thumbnail store with no second mechanism; and the frontend's
`/api/thumb/{hash}` needs no reference-shaped special case. The cost is that
`content_hash` means identity rather than "hash of the bytes".

Two conditions elsewhere read "has no present location" and both would be
catastrophically wrong about a link, since a reference has no location by design.
`is:missing` and `prune` therefore both exclude `kind = 'reference'` explicitly -
without the second, the first `prune` after adding a link deletes every reference in
the library.

## Tagging

Canonical form is kebab-case, resolved through `tag_alias` before insert, so `Pixel Art`,
`pixelart` and `pixel_art` all land on `pixel-art`.

**Heuristic sources**, no models:

- Filename tokenisation: split on separators and camelCase, drop numeric suffixes and
  common noise (`final`, `v2`, `copy`, `export`), so `goblin_walk_01.png` yields `goblin`
  and `walk`.
- Parent folder names up to `folder_depth`, so `.../Characters/Enemies/goblin.png` adds
  `characters` and `enemies`.
- Root name as a `source:` namespaced tag.

**CLIP zero-shot** scores each image against the vocabulary in `tagging/vocab.py`.
Scoring against a fixed vocabulary rather than generating prose is deliberate: it produces
canonical, filterable tags, and a small model does this far better than it writes
captions.

Each namespace is scored as its own softmax - including a "none of these" option that is
never emitted - and only its winner is written, and only above `clip_threshold`. So an
asset gets at most one `type`, one `style` and one `subject`, and often none. The
alternative, keeping every label above a cosine cutoff, is what the spec originally said
and does not survive contact with real scores; M4 below has the numbers.

**VLM captions** are the one tier that writes prose rather than tags, which is why they
are opt-in per asset or per collection and never part of a scan. They exist for the case
tags cannot reach: `prop_04.fbx` in a folder called `Set2` is unfindable until something
looks at it and says "a wooden barrel with metal bands". The caption lands in
`asset.caption` and therefore in FTS, so it is searchable the moment it is written.

`moondream` through the ollama already installed here, which is why this tier needs no
Python dependency: it is one HTTP call, the weights live in ollama's own store, and it
unloads them when idle. What the model is shown is the 256 px thumbnail, re-encoded to
PNG because ollama cannot read WebP. Audio, references and normal maps are never queued -
none of them is a picture of anything, and M5 below has what the model said about them.

Starting vocabulary, extended over time:

```
type:     tileset  character-sprite  ui-icon  weapon  prop  vfx  portrait
          background  font-atlas  normal-map  concept-art  texture
style:    pixel-art  hand-drawn  low-poly  flat  realistic  isometric  top-down
subject:  fantasy  sci-fi  medieval  modern  nature  dungeon  town  cave
```

**Text LLM** (`qwen2.5:3b`, already installed) canonicalises noisy tag candidates and
turns fetched page titles into tags for reference assets.

## Search grammar

Parsed in `search.py` into a filter tree, compiled to SQL plus an optional vector pass.

```
dark cave                  bare words: FTS match, plus semantic pass when CLIP is present
"exact phrase"             FTS phrase
tag:tileset  -tag:wip      include / exclude
kind:image | model3d | audio | reference
license:cc0    source:kenney    root:prototype    collection:jam
w:>=512  h:<64  size:>1mb  tris:<5000  dur:<2s
rate:>48000  channels:1  depth:24  bitrate:>192kbps  peak:>-6db  rms:<-30db
has:alpha | has:animation | has:caption | has:license
is:managed | is:missing | is:untagged | is:vendor
is:clipping | is:silent | is:mono | is:stereo
similar:1234               dHash neighbours, or CLIP neighbours when available
sort:added | name | size | relevance
```

Bare-word results and semantic results merge by normalised score rather than
concatenating, so a strong semantic hit can outrank a weak literal one.

Two units are not what they look like. `db` is a multiplier of one, present only so
`peak:>-6db` reads the way it would be said out loud, and decibels are the reason the
number pattern accepts a leading minus at all - it cannot collide with `-tag:wip`,
which is stripped from the front of a whole token long before a value is parsed. `kbps`
is decimal where `kb` is binary, because a 192 kbps file is 192,000 bits per second and
a threshold that quietly meant 196,608 would exclude the files it was typed to find.

`is:clipping` and `is:mono` are `peak:` and `channels:` underneath and exist anyway.
The useful queries are the ones nobody phrases as a number: "which of these clips" is
the question, and `peak:>=-0.1db` is that question asked backwards.

## HTTP API

```
GET    /api/capabilities              which optional extras are live
GET    /api/assets?q=&sort=&limit=&offset=
GET    /api/assets/{id}
PATCH  /api/assets/{id}               title, notes, source_url, source_name, license
POST   /api/assets/{id}/tags          add manual tags
DELETE /api/assets/{id}/tags/{tag}
POST   /api/assets/bulk/tags          add/remove across an id list
GET    /api/assets/{id}/similar
GET    /api/facets?q=                 tag counts for the current query
GET    /api/thumb/{hash}              webp, or original passthrough when skipped
GET    /api/file/{id}                 raw bytes, backs drag-out
POST   /api/assets/{id}/reveal        open Finder at the file
POST   /api/assets/copy               ids + destination
GET    /api/roots   POST /api/roots   DELETE /api/roots?path=
POST   /api/scan                      start a scan
GET    /api/scan/status               SSE progress stream
GET    /api/collections               POST, PATCH, DELETE
POST   /api/collections/{id}/assets   DELETE the same path to remove
POST   /api/import                    multipart upload into the vault
GET    /api/maintenance               counts that move while a page is open
POST   /api/model/download            fetch the CLIP weights, progress on the SSE
POST   /api/vlm/pull                  fetch the caption model through ollama
POST   /api/embed                     queue embeddings for whatever has none
POST   /api/thumbs                    queue tiles for whatever should have one
POST   /api/prune                     dry run unless {"confirm": true}
```

Plus four the frontend needed that this list did not anticipate: `/api/assets/count`,
so the grid can size its scrollbar before it has fetched a page; `/api/tags?prefix=`
for autocomplete; `PATCH /api/assets/bulk` for the licence and attribution an imported
pack shares; and `POST /api/assets/summary`, which is what lets bulk mode describe a
400-asset selection in one request. The three under `/api/assets/` are declared above
`/api/assets/{asset_id}` - see M3 below for what happens otherwise.

**The last six close a gap that was never deliberate.** `/api/capabilities` reported
precisely which optional piece was missing and then offered no way to go and get it, so
installing the extras - and, worse, registering the very first root - were the workflows
that still assumed a terminal. Two shapes, chosen by what the work is. Fetching a model
is one slow download with byte progress, so it gets a runner and a thread, and reports
on the status stream beside the scan; `FetchRunner` takes both because
`clip.download` and `vlm.pull` already share a `(label, done, total)` progress
signature, and because running two large downloads at once on one laptop helps nobody.
Embedding and thumbnailing are per-asset work the job queue already existed for, so
those only enqueue and nudge, and report through the queue counts that were already
being streamed.

`POST /api/thumbs` does not queue everything without a tile, which is the obvious
reading and the wrong one: with `skip_smaller` on, a 32x32 sprite legitimately has no
file, so that rule re-queues every small sprite in the library on every run and reports
a number it will do nothing with. It mirrors `thumbs._image_thumb`'s own skip rule
instead, from the probed `width`/`height`.

`/api/capabilities` is load-bearing: the frontend hides controls for anything missing
rather than showing buttons that error. Same principle as AssetGeneratorHelper's optional
`segment` extra. It reports `clip`, `clip_weights`, `clip_deps` and `embedded`
separately, because those fail differently and want different things done about them: a
model installed over a library nothing has embedded searches exactly like no model at
all, and telling somebody to download weights they already have is worse than saying
nothing. `vlm` and `vlm_server` are split for the same reason one tier down: no ollama
wants `ollama serve` and ollama without the model wants `assetkeep vlm pull`, and
"unavailable" gives the advice that fixes the other one.

There is no endpoint for downloading the weights. Half a gigabyte over an HTTP request
with no progress channel is the wrong shape for a browser; `assetkeep model download` is
the right one, and the UI's job is to say so. `assetkeep vlm pull` is the same shape for
the same reason.

M5 adds four endpoints:

```
POST   /api/references                {url} -> a reference asset, fetched now
POST   /api/references/{id}/refresh   fetch the page again
POST   /api/collections/{id}/export   destination, layout, folder, manifest
POST   /api/captions                  queue captions for an id list
```

The first two are synchronous and the last is queued, which is the whole difference
between them: one page and its preview is a second or two, and a link that appears only
after a background job has run is a link somebody adds twice. A caption is seconds of a
1.7 GB model's attention, a selection is often forty of them, and the queue and its
progress stream exist for exactly that. `/api/captions` answers `409` rather than
queueing work nothing will run when ollama is not there, and the message is the command
that fixes it.

## Frontend

```
web/
  index.html
  api.js          fetch wrappers, capability cache
  state.js        single store, query is the canonical state
  search.js       grammar parse/serialise, filter chips
  grid.js         virtualised grid
  quicklook.js    fullscreen preview, spritesheet playback, turntable scrub
  inspector.js    slide-over metadata editor
  style.css
```

**The query string is the single source of truth for what is displayed.** Clicking a tag
in the sidebar appends `tag:x` to the query rather than maintaining parallel state, and
every active filter renders as one removable chip. This avoids the usual failure where
facet clicks and typed queries disagree about what is being shown.

Grid rendering: absolutely positioned cells inside a sized scroll container, rendering
only the visible window plus a buffer of two rows. Fixed square cells, `object-fit:
contain`, `image-rendering: pixelated`, CSS checkerboard behind transparency. Cell size
steps 96 / 128 / 192 / 256.

Drag-out sets `e.dataTransfer.setData("DownloadURL", ...)` pointing at `/api/file/{id}`.
Chromium honours this and writes the file to the drop target. Other browsers ignore it, so
it is wired as a bonus on top of Reveal, Copy-to and Copy-path rather than as the primary
retrieval path. A reference drags out as `text/uri-list` instead, since it has no bytes
to hand over and a URL is what an editor or another browser will take.

**Adding a link is the same gesture as adding a file.** The grid takes a drop either way:
a link dragged from another tab arrives as `text/uri-list` rather than as a file, and the
dropzone says which of the two is about to happen. That is the frontend half of the
argument that a reference is an ordinary asset - one drop target, one grid, one
inspector, and `kind:reference` the only thing that distinguishes one.

Two behaviours a reference changes rather than adds. `r` opens the page instead of
revealing a file, because "show me where this is" is one intention with two answers. And
the inspector hides Locations for a link, which is not an empty section but an
inapplicable one.

## Milestones

### M1 - Core index

- [x] `config.py`: load, defaults, write-back, path expansion
- [x] `db.py`: schema, `user_version` migrations, WAL, connection helper
- [x] `roots.py`: registry, glob excludes with defaults merged, symlink policy
- [x] `hashing.py`: chunked blake2b
- [x] `scan.py`: walk, `(size, mtime)` fast path, hash, upsert, absent marking
- [x] `probe/image.py`: attributes plus the six heuristics, PSD and EXR handling
- [x] `probe/model3d.py`: assimp loader chain for FBX, trimesh for the rest,
      sibling-preview lookup
- [x] `probe/audio.py`: stdlib wave, ffprobe fallback, duration heuristics
- [x] `similarity.py`: dHash and palette extraction
- [x] `tagging/heuristic.py` and `tagging/vocab.py`: tokenisation, aliases, canonical form
- [x] CLI: `root add/list/remove`, `scan` (`--rehash`, `--reprobe`), `search`,
      `show`, `tag`, `prune`, `capabilities`
- [x] `search.py`: the full grammar, pulled forward from M2 because the CLI's
      acceptance test needs it

Done when scanning `MyStarterUtils/Assets` yields roughly 900 assets (502 png, 223 fbx,
163 audio, plus stragglers) rather than 5,323, proving the extension allowlist and
excludes both hold, and `assetkeep search 'tag:pixel-art w:>=64'` returns the right files.

**Measured.** The tree has since grown to 11,417 files; the scan indexes 929 of them,
with the predicted breakdown intact: 502 png, 223 fbx, 143 wav plus 20 ogg, 16 psd,
13 exr, 4 obj, 3 tif. Cold scan 35 s, rescan 0.45 s.

Four heuristics needed guards that only real files revealed, each now carrying a
regression test:

- A detected block period is believed only when **every block is measurably flat**.
  Comparing the colour count against the implied logical pixel count reads well and
  does not work, because the count saturates: a 512 px photographic texture on a
  claimed 8 px grid has exactly 4,096 logical pixels, so the test was `4096 <= 4096`
  for every photograph in the library. 202 pixel-art tags fell to 133, and the 48
  claiming a measured grid fell to 21, all of them genuinely in pixel or VFX folders.
- A block must also span at least 8 blocks per axis and divide one axis exactly. A
  48 px sprite was being reported as a 30 px grid.
- `tileable` requires the border to carry a comparable share of the image's own
  contrast. A 2048 px character atlas on a flat background scores a perfect seam
  because both its edges are the same solid colour. 191 tags fell to 110.
- A spritesheet needs two *filled* cells, not just two cells. Half the false
  positives were single sprites with one empty half, splitting cleanly into 1x2.

One trap worth recording: `impasse.errors.AssimpError` derives from `BaseException`,
not `Exception`, so a missing assimp sails through `except Exception` and kills the
scan. Both the probe dispatcher and the loader chain catch `BaseException` and
re-raise control flow.

**With assimp 6.0.5 installed**, 224 of 226 models yield geometry, rigs and animation
counts (836 bones and 3 clips on `Colonel@T-Pose`; 275 bones on `MMExplodude`), adding
`low-poly` to 139 models, `animation` to 141 and `rigged` to 28. The `/opt/homebrew/lib`
hint in the loader chain is what finds it: neither binding searches there, and Apple
Silicon Homebrew installs nowhere else. The two failures are FBX 6.1.0 **ASCII** files
from a 2016 Blender exporter, a format assimp 6 no longer reads. They still index,
hash, tag and search; only the geometry is absent.

That exposed a gap the spec did not anticipate. Probes run on first sight of a hash,
so installing a capability does nothing for content already indexed, and the only
remedy was deleting the index - which also destroys every manual tag, contradicting
the principle the whole `source` column exists to protect. Hence
**`scan --reprobe`**: re-extract attributes and automated tags over existing content
without re-hashing, since the `(size, mtime)` fast path has already proved the bytes
are unchanged. Verified on the real corpus, recovering all 224 models' geometry in
40 s with manual tags intact.

The subtlety worth keeping: automated tags are cleared once per asset per run, not per
file. Per file would be wrong for content in two places, because walking the second
location would wipe the folder tags the first had just written.

### M2 - Browsing UI

- [x] `thumbs.py`: WebP encoder with skip-small and lossless/lossy selection, sharded
      store, numpy rasteriser for 3D, waveform renderer for audio, ffmpeg path for EXR
- [x] `job.py`: background worker draining the queue, SSE progress
- [x] `server.py`: assets, facets, thumb, file, scan, capabilities
- [x] `search.py`: full grammar to SQL (landed in M1)
- [x] Frontend: sidebar, query bar with chips, virtualised grid, Quick Look, keyboard map
- [x] Retrieval: reveal, copy-to, copy-path, drag-out
- [x] `similar:` backed by dHash

Done when browsing several thousand assets is smooth and every keyboard shortcut works.

**Measured.** 921 assets thumbnailed in 30 s into 670 files totalling 6.1 MB - the other
251 hit the skip-small rule and are served as originals. Grid, facets, chips, Quick Look
spritesheet playback and `similar:` all verified against the real library in a browser.

Four bugs that only the real library or a real browser exposed:

- **Scene-graph transforms were being ignored.** Every single-mesh prop has an identity
  transform, so crates and bushes rendered correctly and nothing looked wrong. A rigged
  character is eleven meshes on eleven transformed nodes, and in local space all eleven
  sit at the origin, so it rendered as its body parts stacked into a disc. `bounds_x/y/z`
  were wrong for the same reason, which is how a humanoid was recorded as three times
  wider than tall.
- **`_project` decimates dense meshes by rebinding `faces` locally**, while `render_mesh`
  still looped over the original count - so any mesh above 60,000 triangles indexed off
  the end of every array. Loop over the projected array, not the input.
- **`sqlite3.ProgrammingError` on nearly every request**, in the module whose docstring
  claimed a connection per request prevented it. It does not: a FastAPI sync generator
  dependency has its setup, its endpoint body and its teardown dispatched to *three
  different* threadpool workers. The fix is `check_same_thread=False` on a connection
  that is still never shared between concurrent requests.
- **`similar:` returned nothing**, because a 12-bit cutoff is the figure everyone quotes
  and it is wrong for this content. Measured distances from one sprite to all others run
  13, 17, 18, 18, 19, 19, ... 23, 25 with no gap: the nearest genuine match is at 13 and
  unrelated art blends in around 25. Sprites and tiling textures do not produce the
  bimodal distribution photographs do, so similarity here is a ranking question, not a
  membership one. Now returns the 60 nearest, ranked, with a cutoff only at chance level.

Two smaller ones: Quick Look's spritesheet zoom used `Math.min(1, ...)`, which can only
ever shrink, so every animation played at 1:1 in the middle of a fullscreen view; and the
image probe's colour-count guard could not see past its own saturation cap.

**The frontend has no automated tests.** It was verified by driving headless Chrome over
the DevTools protocol and reading the screenshots, which caught the two UI bugs above.
That is not a substitute for a test suite, and the query grammar in `web/search.js` -
duplicated from `search.py` - is the part most likely to drift.

**The status stream emitted on the wrong condition, and a later headless pass caught
it.** The rule was "emit while busy, plus one frame on going idle", which is not the
same as "emit on change" and differs in exactly one case: an idle stream polls every
three seconds, so a scan that both starts and finishes between two polls is never once
observed as busy. No frame is sent, the grid never learns the scan finished, and it sits
empty until the page is reloaded. That is the ordinary case for a first small root -
the worst possible moment for it, and invisible on the large library the rule was
written against. It now diffs the payload and emits whenever it differs.

Two things about that are worth keeping. Reading it, the old rule looks correct; it
took watching a real browser fail to fill its grid to see otherwise. And the bug was
latent for as long as adding a root meant using the CLI, because whoever did that also
had a terminal telling them the scan had finished.

### M3 - Curation

- [x] Slide-over inspector, editable metadata, source and licence fields
- [x] Tag input with autocomplete over existing tags
- [x] Bulk tag add/remove across a selection
- [x] Collections
- [x] Vault import by drag and drop, including zip extraction

**One panel, two modes.** The inspector shows one asset or the whole selection,
and edits either. A separate bulk screen was the alternative and is the wrong
shape: the selection is what is being worked on in both cases, and a panel that
vanishes when the selection reaches two teaches people to only ever select one.
Bulk mode is one `POST /api/assets/summary` for the whole selection rather than
a fetch per asset, because the interesting selections are the large ones and the
panel has to say something true about 400 assets without 400 round trips. A
field comes back with a value only when every asset agrees; otherwise `null`,
shown as "mixed" and left alone unless it is edited. `title` is refused by the
bulk endpoint outright - it is per-asset by definition, and setting forty to one
string destroys them.

**Only manual tags get a remove button.** Removing an automated one would look
like it worked and then grow back on the next rescan, so the button would be a
lie. This is the `(asset_id, tag_id, source)` primary key showing through to the
UI, and it is why the two kinds of tag are drawn differently.

**The batch tag is written with source `imported`.** A rescan clears every
automated source and re-derives them from the path, which would delete a
batch tag and never bring it back: the fact that these forty files arrived
together as one download is not recoverable from where they ended up. `imported`
is the source that exists for exactly this.

**The vault is an ordinary root**, registered with `mode = "managed"` on first
import and walked by the same scanner. An import path with its own indexing
logic would be a second, quietly divergent definition of what an asset is.
Imports are deduplicated by content hash for the same reason identity is a hash
everywhere else, and a duplicate is *reported* rather than silently skipped,
because "I dropped 40 files and got 3" needs an explanation.

Archive extraction is deliberately narrow. Zip Slip (`../../.zshrc`) is the
reason: an archive member is the one filename in this tool that arrives from
somewhere untrusted. Nested archives are refused too - they buy little and are
the shape a decompression bomb takes - along with anything off the extension
allowlist, and members above a per-file and a per-archive size cap. A pack's
`license.txt` is a real loss and is counted as skipped rather than dropped
quietly.

**Measured.** Verified against the same 921-asset library, driving Chrome over
the DevTools protocol. Four things the real thing exposed:

- **`/api/assets/bulk` returned 422 for every call.** FastAPI matches routes in
  declaration order and `bulk` is a perfectly good string to try to parse as the
  `int` in `/api/assets/{asset_id}`, so the detail route claimed it first and
  rejected the body. It reads as a malformed request and is nothing of the sort.
  The literal routes have to be declared above the parameterised one.
- **A test wrote into the real home directory and passed.** The fixture
  overrode `db_path` and `thumbs_path` and not `vault_path`, so the import went
  to `~/AssetKeep/vault` - and the assertion then read the same default it had
  just polluted, so it was green. `tests/conftest.py` now fails any test that
  touches the default paths at all.
- **Opening the panel left the rightmost column of the grid underneath it.**
  Cells are absolutely positioned, so narrowing the container is not enough;
  the positions have to be recomputed.
- **Bulk mode buried every metadata field under sixty tags.** Thirty-five real
  spritesheets carry sixty distinct tags between them, most on exactly one
  asset. Folded at 24, which is where the counts stop meaning anything.

### M4 - Semantic discovery

- [x] ONNX CLIP loader with weight download into `~/AssetKeep/models/`
- [x] Embedding job, batched
- [x] Semantic pass merged into search scoring
- [x] Zero-shot vocabulary tagging with confidence
- [x] `similar:` upgraded to embeddings when present

**The model is pinned to a commit and verified by sha256.** The weights come from
`Xenova/clip-vit-base-patch32`, transformers.js's export of the ViT-B/32 every
CLIP tutorial uses, at a fixed revision rather than `main`. A branch that moves
under a downloader turns "the same command on two machines" into a coin flip.
Each file is streamed to `<name>.part` and renamed only once its digest matches,
because a truncated ONNX graph loads far enough to produce numbers before it
fails, and numbers from half a model are indistinguishable from numbers.

**The semantic pass is injected, not imported.** `compile_query` takes a callable
that turns a string into ranked ids and knows nothing else about it. So
`search.py` never imports onnxruntime, the ranking is testable with a six-line
fake, and a machine with no model runs the code that shipped in M1 rather than a
version of it with the interesting parts disabled. `vectors.py` is the same
split on the storage side: pure numpy, so everything that *reads* embeddings
keeps working after the optional extra is uninstalled.

**Ranks are fused, not scores.** bm25 returns an unbounded negative number whose
scale depends on the corpus; a cosine returns 0.20 to 0.30 in a band that
depends on the prompt. There is no honest way to put those on one axis, and
every attempt - min-max within the page, a fixed multiplier, a tuned weight -
becomes a constant somebody has to retune whenever either side changes. What
both methods do agree on is what an ordering is, so reciprocal-rank fusion
combines the orderings. An asset in both lists beats one that topped a single
list, which is the behaviour worth having: the two passes fail differently, so
what they agree on is rarely wrong.

The literal side stays exhaustive. Every FTS match is still a result, exactly as
without a model; the semantic side adds its best 120 above a floor. That keeps
the guarantee that installing CLIP cannot lose a result that used to be found,
and it is why the cap is on the semantic side only - FTS answers a yes/no
question and returns a set, while a cosine returns a number for every asset in
the library and never says no.

**3D models are embedded through their thumbnails.** CLIP takes pixels and an
FBX is not pixels. The flat-shaded render is what a person recognises the model
by in the grid, so it is the honest thing to index, and it is what puts a
quarter of the library into semantic search rather than leaving it reachable
only by filename. Audio is left out: its thumbnail is a waveform, and a waveform
embedded as a picture produces a confident and entirely fictional opinion about
what the sound is of.

**Zero-shot tagging scores a softmax within each namespace, not a raw cosine.**
The spec said "keep everything above `clip_threshold`" and that does not work.
Image-text cosines run from about 0.20 to 0.30 with no gap anywhere, and where
in that band a label sits is mostly a fact about the label: "a repeating surface
pattern" scores higher against everything than "a silhouette in white on black"
does, so one fixed cutoff either admits `texture` onto the whole library or
admits `mask` onto none of it. A softmax within the namespace asks the
comparable question - which of these did it prefer, and by how much - and that
answer transfers. `clip_threshold` keeps its name and changes its meaning, from
a cosine to a probability.

Each namespace also carries background rows: descriptions like "an abstract
pattern of colour" that are scored and never emitted. Without them a softmax
over eight subjects has to pick one, and it did: `dungeon` landed on 146 of 708
assets including a selection box and a roughness map, because "a dark stone
dungeon" was the least wrong of eight settings for a grey rectangle. With
somewhere else to put the mass it falls to 13.

**Measured.** Against the same library, 708 of 922 assets embedded - the other
214 are audio, plus 53 animation-only FBX exports carrying one clip and zero
triangles, which render nothing because there is nothing in them to render.

A retrieval benchmark of ten queries, scored against ground truth from the
heuristic taggers, which use no model at all: filenames, folder names,
blue-channel dominance, grid detection. Independent labels are the only reason
this is a benchmark rather than a mirror.

| vision | text | P@10 | MRR |
|---|---|---|---|
| int8 | int8 | 0.260 | 0.402 |
| int8 | fp32 | 0.340 | 0.533 |
| fp32 | int8 | 0.310 | 0.505 |
| fp32 | fp32 | **0.360** | **0.645** |

This reversed the intended default. The quantised pair is 149 MB against 580 and
looked like the obvious choice for an 8 GB machine, right up until it was
measured. Both towers cost something and the text tower costs more, and int8 and
fp32 embeddings of the *same image* agree only to a cosine of 0.94 - which sounds
close and is not, in a space where two unrelated images already sit at 0.74. The
saving was 430 MB of disk, on a machine with 91 GB free, for a worse index that
is expensive to rebuild. Full precision is the default and `clip_model` selects
the other. Embedding 708 assets takes 34 s at full precision and 16 s quantised,
which decides nothing either way.

Preprocessing was argued about first and measured second, and the argument lost:

| preprocessing | P@10 | MRR |
|---|---|---|
| black, letterboxed | **0.360** | **0.645** |
| white, letterboxed | 0.350 | 0.585 |
| grey, letterboxed | 0.330 | 0.548 |
| black, centre-cropped | 0.360 | 0.543 |

Mid grey was chosen on the reasoning that black loses a dark-outlined sprite's
outline and white erases a glow or spark sheet, so grey keeps both readable. It
finished last, at both precisions. Black wins on both metrics, and it is what
`similarity.py` already composites dHash onto, so the two similarity measures
now agree about what a sprite looks like.

Letterboxing survived. CLIP's own preprocessing resizes the short edge and crops
the middle out, which is right for photographs and throws away three quarters of
a 1024x256 sprite strip; the whole image scaled into the square scores the same
P@10 and a distinctly better MRR.

Thresholds are calibrated rather than guessed, because none of these numbers
mean what they look like:

- **Neighbours, 0.80.** Over all 500,556 image pairs, the 1st percentile cosine
  is 0.537 and the median 0.736 - two entirely unrelated sprites score 0.74,
  because the image tower's outputs occupy a narrow cone. The initial 0.55
  admitted the whole library as everything's neighbour. The distribution alone
  argues for 0.85, and looking at what that excludes argues back: for a sword
  model 0.85 returns one neighbour, and the 0.82s it drops are a plant, a crate
  piece and two pillars - every other small low-poly prop there is. At 0.80 the
  median asset has 74 neighbours, `SIMILAR_LIMIT` caps what anyone sees at 60,
  and the 2 percent with none fall back to the perceptual hash.
- **Semantic hits, 0.24.** Image-text scores live in a different band entirely -
  pooled over fourteen realistic queries, the median is 0.223 and the 99th
  percentile 0.271. So 0.24 is roughly "the top one percent": a median of 62
  results per query, no query empty-handed. At 0.28, half the queries return
  nothing at all.
- **Zero-shot, 0.40.** At 0.35 the tagger puts `normal-map` on a fader mask and
  `pixel-art` on a debug menu. At 0.40 both go and everything that was right
  stays. Against the blue-dominance normal-map detector, which shares no code
  and no input with CLIP, the top-1 zero-shot answer agrees on 31 of 33.

### M5 - References and export

- [x] `probe/reference.py`, add-URL flow, og:image thumbnails
- [x] `reference.py`: identity by URL, refresh, site and title tags
- [x] `export.py`: a collection into a project folder, plus manifest and credits
- [x] Optional VLM captions via ollama, per asset or per collection
- [x] CLI: `reference add/refresh`, `export`, `caption`, `vlm status/pull`
- [x] API: `POST /api/references`, `.../refresh`, `.../export`, `/api/captions`
- [x] UI: a Link button and URL drops, link tiles and Quick Look, an export
      action per collection, and a Describe button in the inspector

Done when a URL becomes an asset that behaves like every other asset, a
collection lands in a Unity project with its licensing written down beside it,
and captions can be asked for without being inflicted on the whole library.

**References, measured against six real URLs.** kenney.nl, opengameart.org and
itch.io all yield a title, a description and an `og:image`; a
raw.githubusercontent.com PNG takes the direct-image path and becomes its own
preview. Two of the six fail and both fail usefully: a 404 leaves a reference
titled from its own URL slug (`platformer pack redux`), and
upload.wikimedia.org answers `403` to any client that is not a browser, so the
link is indexed with no preview. The fetch is 1.3 s for a dead link and about
6 s for a page plus its image.

A Kenney page produces the tags `kenney-nl` (as `source:`), `tiny`, `dungeon` -
that last one from the page title, which is the only material a reference has,
and better material than a filename because a person wrote it to be recognised.

**Export, measured on a 128-asset collection.** 126 files into
`UnityTarget/Assets/jam-prototype/` in 0.26 s; the run again copies nothing and
reports 126 unchanged, because content is compared before anything is written.
The two references in the collection are recorded rather than copied.

`CREDITS.md` was the part that needed measuring rather than designing. Written
as one line per asset it is 130 lines whose single line of actual attribution is
invisible, so a group lists twenty and then counts the rest, with the exhaustive
list in `assetkeep.json` beside it. The section heading a real library earns is
**"No source or licence recorded (91)"**, which is the file doing its job: an
export whose licensing is unrecorded is a thing to go and fix, not an absence to
leave implied.

**Captions.** 703 of 928 assets are captionable: 482 images and 221 models, with
audio, references and normal maps excluded by rule. A 122-asset collection took
8 minutes 38 seconds with nothing else talking to ollama, and failed zero times
- 4.2 s an asset, so roughly 50 minutes for the whole library. That is the whole
argument for this tier being opt-in per collection rather than part of a scan.

Four things had to be measured, and three of them reversed a decision:

- **ollama cannot decode WebP.** It answers `400 Failed to load image or audio
  file`, and WebP is every thumbnail this tool produces - so the tier failed on
  every asset with a tile until `vlm.encode` re-encoded to PNG. The thumbnail is
  still the right thing to send: moondream resizes to 378 px internally and
  base64 of a 4K PNG is 20 MB over a socket for nothing.
- **The prompt had to get shorter, not better.** "This is a game art asset.
  Describe what it shows in one short sentence, naming the subject and its
  style" produced `!!!POLYGONAL TREE!!!`, `***************` and `xtremely
  detailed and colorful checkerboard`. "Describe this image." produced "a 3D
  rendering of a tree, composed entirely of geometric shapes" for the same file.
  moondream2 is a captioner behind a Question/Answer template rather than an
  instruction-following model, and a long instruction lands in the question slot
  and derails it. Shaping the answer belongs in `vlm.tidy`, not in the prompt.
- **White backdrop, against CLIP's black.** On black, moondream described a UI
  icon atlas and a selection box as "a black and white photograph of an empty
  room with no visible objects", twice, in those words. On white the same two
  came back as "a grid of black and white icons" and "a square-shaped window".
  CLIP wanted black for the opposite reason: it scores rather than describes,
  and a glow sheet on white scores as nothing. Two models, two backdrops, both
  measured rather than argued.
- **Normal maps are excluded.** "A vibrant purple square", "a hexagonal pattern
  in shades of blue and purple", "a close-up view of a circuit board" - all
  accurate about the pixels and useless about the asset, and worse than useless
  in an index where every normal map would then answer to "purple". Same call
  as the audio exclusion: a file that is not a picture of anything gets a
  confident description of the thing it is not.

Quality, honestly: it is good on anything with a silhouette, weak on grey prop
renders, and confidently wrong often enough to be worth saying out loud.
`2D_Character_Animation_Idle_SpriteSheet` comes back as "a grid of 16 black and
red robots, each with two legs and two arms, arranged in four rows of four",
`turret` as "a military-style gun", `hearts` as "two heart-shaped icons, one red
and the other white", `GrasslandsTrees` as "four distinct trees, each with its
own unique shape and size". Against that, `LoftWeaponIcons` is "a collection of
six crayons", `KoalaPickups` is "six bulletproof vests", `Chest` and `Container`
are both "a gray rectangular object, possibly a box or a container", and
`Wall_Short` is "an open book". For 3D the limit is the flat-shaded render
rather than the model; for the 2D misses it is a 1.8B model guessing at small
stylised icons.

That is an acceptable trade for what this is: words in the index for assets that
had none. A wrong caption costs a few odd search hits, which is recoverable; it
is stored in `asset.caption`, which is an editable field, so a wrong one can be
fixed in the inspector like any other piece of metadata.

**A concurrency finding the queue design had hidden until now.** A run of 172
captions produced 50 failures, every one a `400` from ollama, and every one of
the fifty succeeded when asked again alone. The cause is two AssetKeep processes
draining one queue: a `serve` left running has a worker thread of its own, and
the CLI's `caption` joins it, so two generate requests hit a 1.7 GB vision model
on an 8 GB machine at once. With nothing else talking to ollama the same assets
fail zero times. Thumbnails never exposed this because they are local and cheap.

`vlm.caption` now retries once after two seconds, because the queue's own three
attempts do not cover it - all three happen back to back inside one drain, so a
two-second overload burns the lot. The error message now carries ollama's own
response body as well, since `400 Bad Request` fifty times is a mystery and
`Failed to load image or audio file` is a fix.

One plain bug, found by measuring rather than by testing: `caption --limit 6`
enqueued 161 jobs, printed 6, and then spent six minutes on all of them. The
limit was applied to the reported list rather than to what was queued, and the
queue does not know it was only asked for a few. `enqueue_captions` takes the
limit now.

**Three more the headless run found**, which is what that pass is for:

- `addEventListener("click", addLink)` hands the listener a `MouseEvent` where
  `addLink` expects a URL, so the Link button added a reference to the string
  form of a click. It *worked*, which is the interesting part: `urlparse` reads
  `https://{'isTrusted': true}` as a perfectly good host. `normalise` now
  checks that a host looks like one, and requires a dot when no scheme was
  typed - `http://nas:8080/x` still works, `nas` does not.
- The "Open the page" action is an anchor, and every `.ghost` rule in the
  stylesheet is qualified to `button`, so it rendered as a default underlined
  blue browser link beside a grey button.
- A stale Chrome profile served an old `grid.js` for one run, which showed as a
  missing host line under reference tiles and cost twenty minutes of looking for
  a bug in code that was already right. The driver gets a fresh profile now.

### M6 - Audio, properly

- [x] `probe/audio.py`: bit depth read from the field ffprobe actually fills,
      `bitrate`, and peak/RMS loudness in dBFS
- [x] `decode_pcm` moved into the probe, where decoding audio belongs
- [x] Waveforms at absolute scale with square-rooted amplitude, clipped
      columns in a warning colour, and no truncation of short clips
- [x] Search: `rate:` `channels:` `depth:` `bitrate:` `peak:` `rms:`, and
      `is:clipping | silent | mono | stereo`; negative numbers and `db` /
      `kbps` units in the grammar
- [x] `web/audition.js`: `p` plays the cursor asset and follows the cursor
- [x] Quick Look: a wide waveform with a playhead and click-to-seek
- [x] Inspector, Quick Look and CLI all state rate, channels, depth and level
- [x] `tests/test_audio.py`, the first tests audio has had

Done when a folder of sound effects can be gone through by ear without leaving
the grid, and when the questions a Unity project asks about audio - what is
stereo, what is clipping, what is 96 kHz for no reason - are things the query
bar can answer.

**The library, measured.** 215 audio files across the Unity projects: 155 WAV,
55 OGG, 5 MP3. The three roots scanned as one library index 175 of them
alongside 809 other assets. Durations run from 0.04 s to 262 s with a median of
0.75 s, which is what an asset library of sound effects looks like: 164 of the
215 tag `sfx`, 9 tag `music`, and 42 fall in the deliberate gap between two and
thirty seconds and are left alone.

**Two probe fields were wrong or missing, and measuring is what found them.**
`bits_per_raw_sample` is the obvious field for bit depth and is the wrong one:
ffprobe leaves it unset for every 16-bit PCM WAV, which is the most common audio
format there is. Reading `bits_per_sample` first took coverage from 136 of 215
files to 155 - every WAV in the library, including the 19 float32 ones the
stdlib `wave` module refuses and which had been falling through to ffprobe and
recording nothing. For the 60 lossy files the field is `0` rather than absent,
because the concept does not apply, and `bitrate` is what carries the equivalent
information; that was being discarded entirely and is now recorded for all 215.

The WAV special case also got the measurement it never had: 3.1 ms in-process
against ffprobe's 26 ms, so avoiding one subprocess per file is worth eight
times its complexity on the 72% of a game project's audio that is WAV.

**Peak-normalised waveforms were the real mistake, and it was invisible until
it was measured.** Every audio editor scales a waveform so the file's own
loudest moment fills the frame. In a grid that is wrong, because a grid is a
comparison: 67 of the 215 files peak within a decibel of full scale and 18 sit
more than 24 dB below it, and all 85 were drawn identically. The one question a
wall of waveforms should answer at a glance - why is that one so quiet - was the
one it could not.

Absolute scale fixes that and introduces a second problem: linear amplitude at
-24 dB is 6% of full scale, three pixels of a 256 px tile, which reads as an
empty box rather than as a quiet sound. Square-rooting the amplitude is the
standard half-decibel compromise and it is the right one here - full scale stays
full height, -6 dB draws at 71%, -24 dB at 25%. Measured on the real files, the
quietest five now draw 28 to 42 rows of 256 and the loudest five draw 230 to
236, where before every one of them drew 118.

**A threshold that was defined twice, and disagreed with itself.** The tile
marked a column as clipping when its square-rooted amplitude passed a
hard-coded 0.999, and `is:clipping` matched on peak at or above -0.1 dBFS.
Those are not the same threshold: 0.999 post-square-root works out at -0.017 dB,
so every file between the two - `MMSequencingBass1.wav` at -0.03 dB among them -
matched the search and drew perfectly clean. The tile now derives its threshold
from `audio.CLIPPING_DB`, and across the library all 18 assets that match
`is:clipping` show the warning colour and none of the other 157 do.

This is the failure that argues for the calibration run existing at all. Every
test passed both before and after the fix, because both halves were self
consistent and nothing compared them to each other. What found it was looking at
eighteen tiles that should have been marked and seeing eighteen that were not.

**What the waveforms found in the library.** `LoftDrop.wav`, shipping in the
TopDownEngine demo scenes, is 30 KB of constant -32768: not silence, but every
sample pinned to full negative scale. Peak says full scale and RMS says exactly
the same, and a crest factor of zero is what says there is no sound in the file.
It draws as a solid block of warning colour with no waveform in it at all, which
is the tile doing precisely its job - a broken asset that had been invisible in
a folder listing for as long as the demo has existed.

`is:silent` matched nothing, and that is the correct answer rather than a broken
filter: the quietest file in the library peaks at -36 dB, which is quiet and is
not silent. The two nearest, `LoftWind.wav` and `MMSequencingHat2.wav`, are
legitimately faint rather than empty, and a threshold moved up to catch them
would have called them a name that is not true.

**Cost.** Waveform rendering is 39 ms a file and 695 KB for all 215, a median
tile of 3.3 KB, with zero failures. The loudness measurement adds 36 ms to
probing one audio file against the 3 ms its header costs, which is the one place
this milestone made a scan slower. It is worth it and it is small: audio is 175
of 984 assets, the whole library scans in 56 seconds, and an incremental scan
pays it once per file ever.

**Auditioning is a mode, and that is the whole design.** Quick Look has played
audio since M2, and using it to go through a folder of a hundred and thirty
sound effects is three keystrokes and a full-screen flash per file. `p` starts
an audition on the cursor and then follows the cursor, so arrowing along a row
plays each sound in turn; it ends on `p`, on Escape, or on reaching an asset
that is not audio. That last rule matters more than it sounds - a mode left
armed while you browse a wall of sprites is a mode that surprises somebody
later. Escape unwinds it before the inspector, because a sound playing is the
most recent thing that started.

**The headless run, which is what that pass is for.** Every check passes: the
grid renders 175 waveforms, `p` marks one tile playing and requests one file,
arrowing plays the next, Escape stops it, Quick Look's waveform is 720 px wide
with a playhead that advances and a click a third of the way along that seeks to
0.346 of the duration. Two things it found were in the test script rather than
the tool, and both are worth writing down because they will recur: the query bar
reloads on `input` and not on Enter, so dispatching the wrong event searched for
nothing and then cheerfully checked the unfiltered library; and setting the bar's
value without clearing the existing chip accumulates filters, which is the
documented promote-to-chip behaviour working correctly and looking exactly like
a bug.

## Testing

`pytest`, with `--doctest-modules` over the package as in AssetGeneratorHelper, so the
detection heuristics carry their examples inline.

Synthetic fixtures rather than binary test assets: generate a 4x-upscaled pixel grid, a
3x2 spritesheet with transparent gutters, a seamlessly tiling texture, and a blue-dominant
normal map in the fixture itself. Each heuristic asserts both the positive and a near-miss
negative, since these detectors fail by over-triggering.

Scan correctness gets a temp-directory suite covering the cases that actually break:
a file moved between roots keeps its tags, a duplicate in two locations stays one asset,
an unchanged file is never rehashed, and a vanished file is marked absent rather than
deleted.

**No model in the test suite, and no network either.** The optional tier is tested with a
stub encoder whose embeddings are whatever the test says they are, which is the only way
to test a classifier's behaviour rather than its opinions: against a real model, a test
asserting "this sprite is tagged pixel-art" fails when the model is right and the fixture
is ambiguous, and passes for the wrong reasons the rest of the time. The behaviour that
matters - one tag per namespace at most, nothing when the answer is none of them,
confidence that is a probability, a batch that fails one file without failing fifteen -
is all reachable with a fake. What the real model does is measured against the
calibration library instead, and written down in M4 above.

The same rule holds one tier up and one layer out. The captioning tests stub
`vlm.caption` and assert on which assets are asked about, what happens to the answer, and
that one refusal does not stop the queue; what moondream actually says is in M5 above.
The reference tests stub the two functions in `probe/reference.py` that touch the
network, because whether kenney.nl is up this morning is not a property of this code -
and the parsing is tested against strings, which is what it takes as input anyway.

**And once more for audio, where the installed thing is a codec.** WAV fixtures are real
files written by the stdlib `wave` module, so the header tests exercise the parser rather
than a mock of it. Everything ffprobe answers is tested against the JSON it returns,
which is a string; loudness and waveform drawing are tested against numpy arrays built in
the fixture. Nothing in `tests/test_audio.py` spawns ffmpeg, so nothing in it passes or
fails depending on whether Homebrew has been run on this machine - which matters more
here than elsewhere, because `decode_pcm` returning `None` is a supported state and a
suite that silently skipped it would be testing the wrong branch.

Each threshold carries its near miss, as the image detectors do: `sfx` at 2.0 seconds and
not at 2.1, `is:clipping` at -0.09 dB and not at -0.13. That last pair is the regression
test for the one defect the calibration run found and the suite did not, and it is
written against the gap between the two thresholds rather than against either boundary,
because a test sitting exactly on a boundary is decided by rounding.

`tests/conftest.py` carries three guards, all of them earned. It fails any test that
writes into `~/AssetKeep` or `~/.config/assetkeep`, after one silently did. It
redirects a defaulted `models_path` into `tmp_path`, because reads are the same trap
in reverse: nothing writes there during a test, so the write guard never fires, but a
developer who has downloaded the weights would get a scan that queues embedding jobs
while everybody else gets one that does not.

And it reports ollama absent, whatever the machine is running - the same trap once more,
and this one is not even in the home directory. `test_captions_are_refused_when_ollama_is
_not_there` passed for everybody except the person implementing the feature, because they
were the one with moondream pulled. A suite whose answer depends on what is listening on
localhost is not a suite; a test that wants the tier present says so.
