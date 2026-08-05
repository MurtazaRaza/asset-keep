# Flags

Things raised during development that need a decision or an action from you, or
that are known gaps rather than bugs. Nothing here blocks the next milestone.
Newest first.

## Windows port, 2026-08-05

### A zip could write outside the vault on Windows, and only on Windows

Fixed, no action needed, but worth knowing it was there.

`_archive_members` refused Zip Slip by checking for `..` and `/` in the split
components. Both checks used the local `pathlib.Path`, which is why the hole
only existed on the machine the code had never run on: a member named
`C:/Windows/System32/x.png` splits into three ordinary components on macOS and
into a *drive* plus two on Windows, and joining a drive onto the vault path
replaces it rather than extending it. A leading backslash does the same thing.
Verified against `PureWindowsPath`: the old check let `C:/Windows/System32/x.png`
through and it joined to exactly that. The upload path had the same flaw.

Both now parse with `PurePosixPath` explicitly, so the answer is a property of
the name rather than of the machine reading it. Two tests in `test_vault.py`
cover it and they assert the Windows behaviour from macOS.

### `assetkeep serve` crashed on a machine with no index yet

Fixed. Pre-existing, unrelated to the port, but it is what a fresh clone hits
first so it would have looked like a Windows problem.

With no `index.db` on disk, the job worker thread and the first HTTP request
both open the empty file, both read `user_version` 0, and both run migration 1.
The loser dies on `table root already exists`. Every other entry point connects
once on the main thread before anything else starts, which is why nothing ever
saw it - and why the test suite could not, since its fixture scans first.

The server now creates and migrates the database in its lifespan, before the
worker starts. `test_the_server_creates_the_index_it_is_pointed_at` covers it;
I confirmed it fails without the fix.

### assimp on Windows will need one line of config

Action: yours, once, and only if you want FBX geometry.

There is no winget or choco package for assimp. The three routes that work are
`vcpkg install assimp`, conda-forge, and the installer from the assimp releases
page, and each drops the DLL somewhere different. The usual location for each is
now searched automatically, but if yours lands elsewhere, set `assimp_lib_path`
in the config - the README's Windows section has the exact form.

Without it FBX still indexes, hashes, tags, thumbnails from a sibling preview
and searches. What is missing is triangle counts, bounds and the rig flag.

### CLIP stays on CPU on the Windows machine, deliberately

No action needed, but it is the one place where "better GPU" does not translate
into "use the GPU".

Vectors from two execution providers are not bit-identical, and `similar:`
compares them directly, so a library half-embedded on CUDA and half on CPU would
rank subtly wrongly forever - with nothing to see, since every number involved
stays plausible. Embedding 900 assets takes well under a minute on CPU either
way. If you ever do want the GPU, the honest way is to re-embed the whole
library with `assetkeep embed --redo` on one provider and stay there.

## M6 (audio), 2026-08-05

### The "through M5" commit is missing some of M4 and M5

Action: yours to decide, and nothing to do with M6.

`e29ee68` does not contain `assetkeep/web/settings.js` at all, and holds older
versions of `web/api.js`, `web/index.html`, `web/state.js` and
`tests/test_server.py` - together the settings panel, the roots API, the model
download and prune endpoints, and 155 lines of tests for them. The working tree
has all of it and always did; the commit was taken from a partial snapshot.

I have not committed anything, so `git status` currently shows those four files
plus `settings.js` mixed in with the M6 changes. They are separable: nothing in
their diffs mentions audio, and `git diff` over the four returns zero hits for
`audio`, `waveform`, `audition` or `peak_db`. Splitting them into their own
commit before M6 would put the history back in order.

### I wrote a calibration library into `~/AssetKeep` and then removed it

No action needed. Written down because it is the third time a variant of this
has happened and the guard that exists does not cover it.

I wrote a config for the calibration run with `db_path` and `thumbs_path` at the
top level of the TOML. They belong under `[general]`, so `config.load` never saw
them, fell back to the defaults exactly as designed, and built the whole 984
asset library at `~/AssetKeep/index.db` with 724 thumbnails beside it. I noticed
because a thumbnail I was inspecting did not match what the code produced.

Removed: `~/AssetKeep/index.db`, its `-wal` and `-shm`, and `~/AssetKeep/thumbs`
entirely. `~/AssetKeep/models` was left alone - that is your 580 MB of CLIP
weights and none of my business. The index was 5.6 MB of purely derived data
that a rescan rebuilds in 56 seconds, and the file it overwrote was the 151 KB
empty one the M4 note below already recommended deleting, so nothing was lost.
`~/AssetKeep` now holds `models/` and nothing else.

Worth noting what did not help. `tests/conftest.py` fails any test that writes
under `~/AssetKeep`, and that guard is why the suite is trustworthy - but this
was me running the CLI by hand, which is the same gap the M4 note identified and
which is still open. The thing that would actually close it is `config.load`
warning about top-level keys it does not recognise, since a silent fallback to
defaults is indistinguishable from a config that was read.

### `is:silent` matches nothing in a real library, which is correct

No action needed, but worth knowing before you conclude it is broken.

The threshold is peak below -40 dBFS. The quietest file in the calibration
library is `LoftWind.wav` at -36 dB, so nothing matches. That is the filter
being right: -36 dB is faint, not empty, and moving the threshold up to catch it
would attach a word to two files that are working as intended. What `is:silent`
is for is a broken export, and this library does not contain one.

It does contain the neighbouring case. `LoftDrop.wav`, in the TopDownEngine
demos, is 30 KB of constant -32768 - every sample at full negative scale. It is
found by `is:clipping`, not `is:silent`, and its tile is a solid block of warning
colour with no waveform in it.

### 102 of 175 sounds are stereo clips under two seconds

Action: worth a look, at your convenience, and it is a Unity import setting
rather than anything this tool can fix.

`kind:audio is:stereo dur:<2s` is 102 files. A stereo clip cannot be positioned
in 3D without Force To Mono, and for a sound effect the second channel is bytes
and memory spent on something the spatialiser will discard. Six more are at
96 kHz (`rate:>48000`), which is bandwidth nothing in a game will hear.

All of it is third-party pack content, so this is not a criticism of anything
you did - it is the first time the library has been able to answer the question
at all.

## M5 (references and export), 2026-08-05

### moondream is now pulled into ollama, 1.7 GB

Action: `ollama rm moondream` if you want it gone. Nothing else in the tool
needs it, and `assetkeep vlm pull` fetches it again on demand.

This is the judgement call the M4 note below predicted, and I made it the way
that note concluded I should have the first time: the milestone was authorised,
the tier cannot be measured without the weights, and the download goes to
ollama's own store beside the two models you already have rather than anywhere
this project owns. `~/.ollama/models` went from 3.5 GB to 4.9 GB.

It earned its place. The measurements it produced reversed three decisions -
the prompt, the backdrop, and whether normal maps should be captioned at all -
and each of those is written up in IMPLEMENTATION.md under M5. Two of them would
have shipped wrong: the carefully-worded prompt makes moondream answer
`!!!POLYGONAL TREE!!!`, and the CLIP-matching black backdrop makes it describe a
sprite sheet as "a photograph of an empty room".

### Two AssetKeep processes share one job queue, and captions notice

No action needed. Worth knowing before you leave a server running and then run
`assetkeep caption` in a terminal.

A running `assetkeep serve` drains the background queue in a worker thread, and
the CLI's `caption` drains it too. That has been true since M2 and never
mattered, because thumbnails are local and cheap. A 1.7 GB vision model on an
8 GB machine is neither: two generate requests at once, and ollama refuses one
with a `400`. Measured, that was 50 failures out of 172 captions, and all fifty
succeeded when asked again alone.

`vlm.caption` now retries once after two seconds, which covers it, and the job's
error message now carries ollama's own words rather than the bare status code.
The underlying design - two processes, one SQLite queue, no lease - is unchanged
and still fine for every other job kind. If captioning ever wants to be
parallel-safe properly, that is a `next_attempt_at` column and a migration.

### The credits file names what has no licence recorded

Action: worth a look, at your convenience.

Exporting a 128-asset collection from the calibration library produced a
`CREDITS.md` whose largest section is **"No source or licence recorded (91)"**.
That is the file working as intended rather than a bug, and the 91 are real: 126
files from a Unity project where nothing has been attributed yet. The inspector's
bulk edit is what fixes it - select a pack, set the source and licence once.

## M4 (semantic discovery), 2026-08-05

### Stray `~/AssetKeep/index.db` needs deleting by hand

Action: `rm ~/AssetKeep/index.db` if you want it gone. Not `rm -rf ~/AssetKeep`,
which this file said before the CLIP weights were moved in below it, and which
would now take 580 MB of verified download with it.

I ran `assetkeep model status` against a config path that did not exist. It fell
back to the built-in defaults, and the default `db_path` is `~/AssetKeep/index.db`,
so opening the database created one. The file is 151 KB, schema only, no assets.

It is at the tool's own default path and is harmless if left, but it is not
something you asked for, and it will be picked up as the real index by any
future command run without an explicit `--config`. My own attempts to remove it
were blocked by the sandbox, so it needs to be your `rm`.

Worth noting the guard that exists for this in tests already: `tests/conftest.py`
fails any test that writes into `~/AssetKeep` or `~/.config/assetkeep`. That
guard covers the suite, not me running the CLI by hand, which is exactly how
this happened.

### CLIP weights were in the session scratchpad, not `~/AssetKeep/models`

Resolved 2026-08-05. No action needed. Kept for the reasoning, since the same
judgement call will come up again in M5 if a VLM gets downloaded.

I put 726 MB of ViT-B/32 ONNX weights in the scratchpad rather than the tool's
default `models_path`, on the grounds that a download that size persisting in
your home directory should be your choice. That was the wrong call: you had
already authorised the milestone, and the tool's own default is the right home
for the tool's own weights. Downloading to a location the project cannot see
also meant the work was only reproducible inside one session.

The fp32 pair and the tokenizer are now at `~/AssetKeep/models/clip-vit-base-patch32`,
verified by sha256 against the manifest in `assetkeep/tagging/clip.py` rather
than by size alone, which is all `installed()` checks. `model status` reports
them present, and a semantic search returns the same results it did from the
scratchpad. The int8 pair was left behind: it lost the benchmark and is not the
default.

Weights are deliberately not repo content, and that part was right. They are
derived and re-fetchable, they are pinned to a commit sha and hash-verified so
committing them buys no reproducibility the pin does not already give, and the
package can be installed read-only into `site-packages` where nothing may be
written. They belong beside `index.db`, `thumbs/` and `vault/`.

### The frontend still has no automated tests

No action needed yet. Carried forward from M2 and M3, and now larger.

`assetkeep/web/` is verified by driving headless Chrome over the DevTools
protocol. That found the real UI bugs in M2 and M3, but it is a person deciding
what to check, not a suite that runs on every change.

The specific risk is `web/search.js`, which reimplements the query grammar that
`search.py` owns, so the two can disagree silently. M4 widened that gap: the
grammar now includes a semantic pass and `similar:` has two different meanings
depending on whether embeddings exist, and none of that is checked from the
browser side. A shared grammar fixture that both implementations parse would
close it.
