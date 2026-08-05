# AssetKeep

A local asset index for game development. Point it at folders, it extracts what
it can, tags what it can, and makes everything findable later. No server, no
subscription, no cloud.

Design rationale is in [PLAN.md](PLAN.md); the buildable spec is
[IMPLEMENTATION.md](IMPLEMENTATION.md).

## Status

**M1 (core index), M2 (browsing UI), M3 (curation), M4 (semantic discovery) and
M5 (references and export) are complete.** Config, schema, incremental scan,
image/3D/audio probes, perceptual hashing, heuristic tagging, the search
grammar, a CLI, thumbnails, a background job queue, an HTTP API, a browser UI
with a virtualised grid, sidebar facets, Quick Look and all four retrieval
paths, plus a slide-over inspector, tag autocomplete, bulk editing, collections,
drag-and-drop import into the vault, optional CLIP embeddings behind semantic
search, semantic similarity and zero-shot tagging, and now URLs as assets,
collection export with its licensing written down beside it, and optional
captions from a local vision model.

Deferred by choice: the Obsidian bridge, sync, and audio beyond structural
metadata.

## Install

```bash
uv sync                      # base: Pillow, numpy, stdlib
uv sync --extra model3d      # trimesh for GLTF/GLB/OBJ/STL/PLY, impasse for FBX
uv sync --extra clip         # onnxruntime for semantic search and tagging
```

Captions need no extra at all: that tier talks to a local ollama over HTTP, and
`httpx` is already a base dependency.

Two optional system dependencies, both detected at runtime and both reported by
`assetkeep capabilities`, which also prints the install command for the machine
it is running on:

```bash
brew install assimp          # macOS: FBX, DAE and BLEND parsing
brew install ffmpeg          # macOS: OGG, MP3, FLAC and AIFF metadata
```

```powershell
winget install Gyan.FFmpeg   # Windows: same, and the waveform tiles
vcpkg install assimp         # Windows: see the Windows section below
```

Without assimp, FBX files are still indexed, hashed, tagged and searchable; only
the geometry numbers are missing. Nothing here is required for the tool to run.

Installing one of these later does not require rebuilding the index. Probes only
run on first sight of a hash, so run `assetkeep scan --reprobe` once afterwards
to re-extract attributes and automatic tags over content already indexed. Manual
tags are left alone.

Known limitation: assimp 6 cannot read the pre-7.x ASCII FBX format that old
Blender exporters produced. Those files still index, hash and search; they just
have no geometry. Two of 226 models in the calibration project are affected.

## Use

```bash
uv run assetkeep root add ~/UnityProjects/Prototype/Assets --vendor
uv run assetkeep scan
uv run assetkeep serve               # the browser UI, on :8765
```

The UI generates thumbnails in the background as it runs. To do it up front
instead, or without the server:

```bash
uv run assetkeep thumbs
```

Everything the UI does is also on the CLI:

```bash
uv run assetkeep search 'tag:pixel-art w:>=64'
uv run assetkeep show 412
uv run assetkeep tag 412 goblin enemy
uv run assetkeep set 412 --license cc0 --source Kenney --source-url https://kenney.nl
uv run assetkeep reference add https://kenney.nl/assets/tiny-dungeon
uv run assetkeep export jam-prototype ~/UnityProjects/Prototype/Assets
uv run assetkeep caption --collection jam-prototype
uv run assetkeep capabilities        # which optional extras are live
```

### Semantic search

Optional, and off until you ask for it. With `uv sync --extra clip`:

```bash
uv run assetkeep model download      # 580 MB of CLIP ViT-B/32, once
uv run assetkeep embed               # about 35 s for 700 assets
```

After that, bare words run a semantic pass alongside the full-text one and the
two rankings are merged, so `a wooden crate` finds furniture models that say
nothing about crates, and `fire` finds a torch sprite named `DungeonTorches`.
Every result the text index would have found is still there - the model adds
results and reorders them, and can never remove one.

Two other things turn on at the same time. `similar:412` starts answering "is
this the same sort of thing" rather than "does this look like that", falling
back to the perceptual hash for anything without an embedding. And each image
picks up tags from a fixed vocabulary, scored rather than written: at most one
`type`, one `style` and one `subject`, each with a confidence, each recorded
with source `clip` so a rescan regenerates them and your own tags survive.

3D models are embedded through their thumbnails, since CLIP takes pixels and an
FBX is not pixels. Audio is left out: its thumbnail is a waveform, and a
waveform embedded as a picture yields a confident opinion about a sound that
nothing ever listened to.

```bash
uv run assetkeep model status        # what is installed, and how much is embedded
uv run assetkeep embed --redo        # re-embed everything, after changing models
```

None of this is required. Without it, search is the full-text index, `similar:`
is the perceptual hash, and tagging is the heuristics - which is what M1
shipped, and it works.

### References: the assets that are a URL

An asset store page, a tutorial, an artist's post you keep going back to. Drag a
link onto the grid, press the **Link** button, or:

```bash
uv run assetkeep reference add https://kenney.nl/assets/tiny-dungeon --tag to-buy
```

The page is fetched once for its title, description and `og:image`, and the
image becomes the tile. After that a reference is an ordinary asset: same tags,
same collections, same inspector, same search, with `kind:reference` the only
thing that tells one apart. Pasting a link straight to an image works too - the
image is its own preview.

A page that will not answer costs you a title, not the entry. The reference is
still made, named after its own URL slug, and `assetkeep reference refresh` will
try again later. A refresh fills in what is missing and leaves what is there
alone, because renaming a link to something you will recognise is a normal thing
to do and a refetch that reverted it would be a button nobody presses twice.

References have no file, and two things in the tool read "no file" as "gone":
`is:missing` and `prune` both exclude them explicitly. Without the second, the
first prune after adding a link would delete every link in the library.

### Export: getting a set back out

```bash
uv run assetkeep export jam-prototype ~/UnityProjects/Prototype/Assets
```

Copies the collection into a subfolder named after it, and writes two files
beside it: `assetkeep.json` for a machine and `CREDITS.md` for a person. The
second is the point. A folder of PNGs cannot carry attribution, the source and
licence fields are free to fill in at import and unreconstructable a year later,
and an export whose licensing is unrecorded gets a heading that says so rather
than an absence you might not notice.

Nothing is overwritten and nothing is duplicated: a file already there with the
same content is left alone and reported as unchanged, so re-exporting a
collection that gained two assets copies two assets. A file with the same *name*
and different content gets `-1`. References are listed rather than copied,
because a link has no bytes.

`--layout kind` splits into `Images/`, `Models/` and `Audio/`; `--folder ""`
exports straight into the destination.

### Captions from a local vision model

Optional, opt-in per asset or per collection, and never part of a scan. It needs
ollama running and one model:

```bash
uv run assetkeep vlm status                 # what ollama is holding
uv run assetkeep vlm pull                   # ~1.7 GB of moondream, once
uv run assetkeep caption --collection jam-prototype
```

This is for what tags cannot reach: `prop_04.fbx` in a folder called `Set2` is
unfindable until something looks at it and says "a wooden barrel with metal
bands". Captions go into the full-text index, so they are searchable the moment
they are written, and `has:caption` finds what has one.

About 3.5 seconds each on an M1 Air, which is roughly 40 minutes for a
900-asset library - hence per collection rather than per library. In the UI it
is the **Describe** button in the inspector, or `d` on a selection, and the
work goes through the same background queue as thumbnails.

Audio, references and normal maps are never captioned. None of them is a picture
of anything, and the model will confidently describe them anyway: a normal map
comes back as "a vibrant purple square", which is accurate about the pixels and
would make every normal map in the library answer to "purple".

It is good on anything with a silhouette - spritesheets, characters, vehicles,
photographic textures - and weak on flat-shaded grey props, where `Chest` and
`Container` both come back as "a gray rectangular object, possibly a box or a
container". The limit there is the render, not the model.

### Getting loose content in

Most assets already live somewhere and get indexed in place. For the rest - the
itch.io bundle still sitting in Downloads, the sprite sheet someone sent you -
there is the vault: drag files, folders or zips onto the grid, or

```bash
uv run assetkeep import ~/Downloads/KenneyPlatformer.zip --collection jam
```

Archives are expanded rather than stored, the same content is never imported
twice, and everything from one import is tagged with the name of that import, so
`tag:kenney-platformer` finds the pack later however deeply the zip nested it.
The vault registers itself as a managed root on first use, which is all
`mode = "managed"` means: this tool may write inside that one folder.

Only indexable extensions come across. A pack's `license.txt` is reported as
skipped rather than dropped quietly, because there is nothing in the index that
could show it.

### Collections

A collection is a set you made by hand, and it is the one thing here that a
rebuild cannot re-derive. Names are canonicalised the same way tags are, so
clicking one in the sidebar and typing `collection:jam-prototype` are the same
operation.

```bash
uv run assetkeep collection new "Jam Prototype"
uv run assetkeep collection add jam-prototype --query 'tag:tileset w:>=512'
uv run assetkeep collection list
```

Deleting a collection never touches the assets in it.

`--vendor` marks a root as third-party pack content rather than your own work,
which is what `is:vendor` and `-is:vendor` filter on later.

### Search grammar

```
dark cave                  bare words, matched against title, notes, filename and
                           tags, plus a semantic pass when CLIP is installed
"exact phrase"             phrase match
tag:tileset  -tag:wip      include / exclude
kind:image | model3d | audio | reference
license:cc0  source:kenney  root:prototype  collection:jam
w:>=512  h:<64  size:>1mb  tris:<5000  dur:<2s
rate:>48000  channels:1  depth:24  bitrate:>192kbps  peak:>-6db  rms:<-30db
has:alpha | animation | rig | caption | license
is:managed | missing | untagged | vendor | clipping | silent | mono | stereo
similar:1234               CLIP neighbours, or perceptual-hash ones
sort:added | name | size | relevance
```

The audio filters answer questions a Unity project actually asks. `is:stereo
dur:<2s` is every sound effect that cannot be positioned in 3D without Force To
Mono, which in the calibration library is 102 of 175 files. `rate:>48000` finds
bytes spent on bandwidth nothing will hear. `is:clipping` finds the 18 files
that touch full scale, one of which turned out to be 30 KB of constant DC that
had been shipping in a demo scene.

Unknown prefixes are searched as words rather than rejected, so a mistyped
filter still returns something.

Filters filter and words rank. A semantic hit is not exempt from `tag:` or
`kind:`; what the model changes is which assets are candidates for the words you
typed, and in what order. Bare words are ordered by the merged ranking unless
you ask for something else with `sort:`.

### Keyboard

Browsing-first, so navigation never needs the mouse.

```
/       focus search          s       find similar to selection
arrows  move in grid          r       reveal in the file manager, or open a link
space   quick look            c       copy to destination
enter   inspector             C       copy path to clipboard
i       inspector             a       select all
esc     unwind one layer      d       describe with the vision model
[       toggle sidebar        +/-     thumbnail size
p       audition audio
```

`p` starts auditioning the sound under the cursor, and keeps going: while it is
on, moving the cursor plays whatever it lands on, so arrowing along a row of
footsteps plays them in turn. It ends on `p` again, on Escape, or on reaching
anything that is not audio. Quick Look can play a sound too, but it is the wrong
tool for going through a hundred of them - that is three keystrokes and a screen
flash per file.

Escape unwinds exactly one layer per press: the field, then Quick Look, then the
audition, then the inspector, then the selection. Closing all of them at once is
how you lose a selection that took a minute to build because a preview happened
to be open.

Clicking a sidebar tag adds `tag:x` to the query; right-clicking adds `-tag:x`.
Every active filter is a removable chip, and the query string is the only state
there is, so the URL is shareable and the back button works.

Getting assets out: **Reveal** always works - Finder, Explorer or whatever file
manager you have - **Copy to destination** remembers the last folder, **Copy
path** goes to the clipboard, **Export** on a collection row writes the whole
set plus its credits into a project, and **drag-out** works in Chromium only (it
needs the `DownloadURL` drag type), which is why it is a bonus on top of the
others rather than the primary path.

## Configuration

`~/.config/assetkeep/config.toml`, created on the first `root add`. See
[config.example.toml](config.example.toml) for every key with its default.

Roots live in this file rather than only in the database, because the database
is derived and disposable: deleting `~/AssetKeep/index.db` and rescanning is a
supported recovery path, and the list of folders to scan could not be rebuilt
from anything if it lived only there.

`~` is the home folder on every platform, so on Windows the config is at
`C:\Users\you\.config\assetkeep\config.toml` and the index at
`C:\Users\you\AssetKeep\index.db`. Write paths in the file with forward slashes -
`"C:/Users/you/Projects"` - because a backslash starts an escape sequence in
TOML and `"C:\Users\..."` is not a valid string. Everything the tool writes back
is already in that form.

## Windows

The tool runs the same on macOS, Windows and Linux. Four things differ, and all
four are things the platform decides rather than choices made here.

**ffmpeg** is `winget install Gyan.FFmpeg`, and it is what audio metadata beyond
WAV, the waveform tiles, the loudness numbers and EXR decoding all need. Open a
new terminal afterwards so the new `PATH` is picked up, then check with
`uv run assetkeep capabilities`.

**assimp** has no package manager entry, which is the only genuinely awkward
part. Three routes work: `vcpkg install assimp`, `conda install -c conda-forge
assimp`, or the installer from the
[assimp releases page](https://github.com/assimp/assimp/releases). All three end
with a DLL - typically `assimp-vc143-mt.dll` - and the usual locations for each
are searched automatically. If yours landed somewhere else, say so once:

```toml
[general]
assimp_lib_path = "C:/vcpkg/installed/x64-windows/bin/assimp-vc143-mt.dll"
```

Either the file or the folder holding it works. Without assimp, FBX still
indexes, hashes, tags, thumbnails from a sibling preview and searches - what is
missing is triangle counts, bounds and the rig flag.

**Long paths.** A Unity project with a deeply nested pack inside it will exceed
the 260-character limit, and a scan skips what it cannot open rather than
failing. Turn the limit off once, from an elevated PowerShell:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

**Reveal** opens Explorer with the file selected, the same as Finder on macOS.
On Linux it opens the containing folder instead: selecting one item is a
per-file-manager flag, and `xdg-open` on the parent is the thing that works
everywhere.

Everything else - the vault, drag-and-drop import, the search grammar, CLIP,
captions through ollama - is unchanged. The one thing worth knowing if you index
the same library from two machines: CLIP runs on CPU deliberately, including on
a machine with a GPU that could do it faster, because vectors from two different
execution providers are not bit-identical and `similar:` compares them directly.
Embedding 900 assets takes well under a minute either way; a library half
embedded on CUDA and half on CPU would rank subtly wrongly forever.

## What gets indexed

An allowlist of extensions, not a blocklist. In the calibration project that
turns 11,417 files into 929 assets without a single hand-written exclude rule.

| Kind | Extensions |
|---|---|
| image | png jpg jpeg webp gif bmp tga tif tiff psd exr aseprite ase |
| model3d | fbx gltf glb obj stl ply dae blend |
| audio | wav ogg mp3 flac aiff aif |

`Library/`, `Temp/`, `Obj/`, `Build/`, `Logs/`, `.git/`, `node_modules/`,
`.DS_Store` and `*.meta` are excluded from every root unless it sets
`exclude_defaults = false`.

## Behaviour worth knowing

**Identity is the content hash, not the path.** A moved file is silently
re-linked on the next scan and keeps every tag. The same file in two places is
one asset with two locations. A reference is hashed on its URL instead, which is
what makes adding the same link twice one asset rather than two.

**Nothing is deleted by a scan.** A file that has vanished has its location
marked absent. Assets with no present location surface under `is:missing`, and
removal is an explicit `assetkeep prune`.

**A file is only hashed when `(size, mtime)` says it changed.** A cold scan of
929 assets takes about 35 seconds; the rescan takes under half a second.

**Probes run on first sight of a hash, not on every scan.** That is what makes
the rescan fast, and it means a newly installed capability needs an explicit
`scan --reprobe` to be applied retroactively.

**Automatic tags are regenerable, manual ones are not touched.** Every tag
records where it came from, and the automated sources are replaced wholesale on
a rescan while `manual` and `imported` rows survive by construction. This shows
through to the UI: only manual tags have a remove button in the inspector, since
removing a derived one would look like it worked and then grow back on the next
rescan.

**Zero-shot tags are a scored choice, not a list of guesses.** Each namespace is
scored as a whole, including an explicit "none of these" option, and only the
winner is written and only if it clears `clip_threshold`. So an asset gets at
most three CLIP tags and often none, which is the point: a tagger that labels
everything has told you nothing. A rescan clears them and the next `embed`
re-derives them from the stored vectors without re-running the model.

**A collection and a manual tag are the only things a rebuild loses.**
Everything else in the index comes back from the files plus the config, which is
what makes deleting `index.db` a supported recovery path rather than a disaster.

## Tests

```bash
uv run pytest
```

Doctests run over the package alongside the suite in `tests/`. Image fixtures
are generated rather than committed, and every detection heuristic is tested
against both a positive and a deliberate near miss, because these detectors fail
by over-triggering rather than by missing things.

`tests/conftest.py` fails any test that writes into `~/AssetKeep` or
`~/.config/assetkeep`. Every path in `Config` has a working default there, which
is right for the tool and a trap for a fixture: one that overrides `db_path` and
forgets `vault_path` pollutes the real home directory and then passes, because
it asserts against the same default it just wrote to. That happened. It also
redirects a defaulted `models_path` into the temp directory, which is the same
trap in reverse - nothing writes there, so the write guard never fires, but a
machine with the CLIP weights downloaded would otherwise run different tests
from one without them.

It also reports ollama absent regardless of what is running on this machine,
which is the same trap a third time: the test asserting that captions are
refused without ollama passed for everyone except the person who had just pulled
the model.

No model runs in the suite, and nothing in it touches the network. The optional
tiers are tested against stubs - an encoder whose embeddings are whatever the
test dictates, a vision model that answers with the filename it was shown, a
page fetch that returns whatever the test says the page said - because a test
asserting that a real model tags a synthetic fixture "pixel-art" fails when the
model is right and passes for the wrong reasons the rest of the time, and
whether kenney.nl is up this morning is not a property of this code. What the
real models do was measured against the 928-asset library and written down in
[IMPLEMENTATION.md](IMPLEMENTATION.md).

The frontend has no automated tests. It was verified by driving headless Chrome
over the DevTools protocol, which is how the UI bugs in M2 and M3 were found,
but that is not the same thing as a suite. `web/search.js` duplicates the query
grammar from `search.py` and is the piece most likely to drift.
