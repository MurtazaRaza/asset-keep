# Flags

Things raised during development that need a decision or an action from you, or
that are known gaps rather than bugs. Nothing here blocks the next milestone.
Newest first.

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
