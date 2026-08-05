// The settings panel: the folders that get indexed, and the optional extras.
//
// Its own module for the same reason the inspector is one - it owns a slab of
// DOM and a set of actions nothing else touches. A modal rather than a sidebar
// section because these are decisions made rarely and read carefully, not
// filters clicked in passing.
//
// Everything here was a terminal command first, and the split was never
// deliberate: `/api/capabilities` already reported exactly which piece was
// missing and then offered no way to go and get it. Each action is a config
// write or a queue nudge, and both report through something that already
// exists - the roots list re-reads from the server, the slow work arrives on
// the status stream.

import * as api from "./api.js";
import { state, refreshRoots, refreshMaintenance, reload } from "./state.js";

let elements = {};
let hooks = {};

export function init(nodes, callbacks = {}) {
  elements = nodes;
  hooks = callbacks;

  elements.close.addEventListener("click", close);
  elements.root.addEventListener("click", (event) => {
    if (event.target === elements.root) close();
  });

  elements.rootAdd.addEventListener("click", () => addRoot());
  elements.rootPath.addEventListener("keydown", (event) => {
    if (event.key === "Enter") addRoot();
  });

  elements.clipDownload.addEventListener("click", downloadWeights);
  elements.embed.addEventListener("click", () => runEmbed(false));
  elements.embedRedo.addEventListener("click", () => runEmbed(true));
  elements.vlmPull.addEventListener("click", pullVlm);
  elements.thumbs.addEventListener("click", rebuildThumbnails);
  elements.retry.addEventListener("click", retryFailed);
  elements.prune.addEventListener("click", runPrune);
}

export const isOpen = () => Boolean(elements.root) && !elements.root.hidden;

export function open() {
  elements.root.hidden = false;
  render();
  refreshMaintenance().catch(() => {});
  elements.rootPath.focus();
}

export function close() {
  elements.root.hidden = true;
}

export function toggle() {
  if (isOpen()) close();
  else open();
}

// --- rendering --------------------------------------------------------------

// Cheap enough to re-run on every status frame, and that is what keeps the
// progress line moving while the worker drains behind the panel.
export function render() {
  if (!isOpen()) return;
  renderRoots();
  renderExtras();
}

//: The scan and queue state this last refetched for, so settling once does not
//: mean refetching on every subsequent frame.
let lastSettled = null;

// Counts come from /api/maintenance, and nothing else refetches it. Rendering
// alone is not enough: a scan started from this panel finishes behind it, and
// without this the numbers keep describing the library as it was at the moment
// the panel opened - which is exactly when they are being read.
export function noteProgress(status) {
  if (!isOpen()) return;
  render();

  const scan = status.scan || {};
  const queue = status.queue || {};
  if (scan.running || queue.pending) return;

  const marker = `${scan.finished || ""}|${queue.done || 0}|${queue.failed || 0}`;
  if (marker === lastSettled) return;
  lastSettled = marker;
  refreshMaintenance().catch(() => {});
}

function renderRoots() {
  elements.roots.replaceChildren(
    ...state.roots.map((root) => {
      const row = document.createElement("li");

      const name = document.createElement("span");
      name.className = "name";
      name.textContent = root.name;
      name.title = root.path;
      row.appendChild(name);

      const flags = [];
      if (root.vendor) flags.push("vendor");
      if (root.mode !== "indexed") flags.push(root.mode);
      if (!root.enabled) flags.push("disabled");
      if (flags.length) {
        const note = document.createElement("span");
        note.className = "ns";
        note.textContent = flags.join(", ");
        row.appendChild(note);
      }

      const count = document.createElement("span");
      count.className = "count";
      count.textContent = `${root.count.toLocaleString()}`;
      row.appendChild(count);

      const drop = document.createElement("button");
      drop.className = "rowdrop";
      drop.textContent = "×";
      drop.title = `Stop indexing ${root.path}`;
      drop.addEventListener("click", () => removeRoot(root));
      row.appendChild(drop);

      const path = document.createElement("small");
      path.className = "path";
      path.textContent = root.path;
      row.appendChild(path);
      return row;
    }),
  );

  elements.rootsEmpty.hidden = state.roots.length > 0;
}

function renderExtras() {
  const caps = state.capabilities;
  const info = state.maintenance;
  const fetching = state.status.fetch || {};

  // Semantic search. Three states that want different things done about them:
  // no runtime is a terminal install, no weights is a download, and a model
  // over an unembedded library searches exactly like no model at all.
  if (!info.clip_deps) {
    elements.clipState.textContent = "runtime missing - uv sync --extra clip";
    elements.clipDownload.hidden = true;
  } else if (!info.clip_weights) {
    const size = megabytes(info.clip_download_bytes);
    elements.clipState.textContent = `${info.clip_label || "weights"} - ${size} to download`;
    elements.clipDownload.hidden = false;
  } else {
    elements.clipState.textContent = `${info.clip_label || info.clip_model} installed`;
    elements.clipDownload.hidden = true;
  }

  const embedded = info.embedded || 0;
  const outstanding = info.outstanding || 0;
  elements.embedState.textContent = outstanding
    ? `${embedded.toLocaleString()} embedded, ${outstanding.toLocaleString()} outstanding`
    : `${embedded.toLocaleString()} embedded`;
  // Nothing to embed with is the only reason to hide the button; nothing left
  // to embed is a reason to disable it, which says something different.
  elements.embed.hidden = !info.clip_weights || !info.clip_deps;
  elements.embed.disabled = outstanding === 0;
  elements.embedRedo.hidden = elements.embed.hidden || embedded === 0;

  // Captions.
  if (!caps.vlm_server) {
    elements.vlmState.textContent = "ollama is not running - ollama serve";
    elements.vlmPull.hidden = true;
  } else if (!caps.vlm) {
    elements.vlmState.textContent = `${caps.vlm_model} is not installed`;
    elements.vlmPull.hidden = false;
  } else {
    elements.vlmState.textContent =
      `${caps.vlm_model} ready - ${(info.captioned || 0).toLocaleString()} captioned`;
    elements.vlmPull.hidden = true;
  }

  // Maintenance.
  const queue = info.queue || {};
  elements.thumbsState.textContent = `${(info.assets || 0).toLocaleString()} asset(s) indexed`;
  elements.queueState.textContent = queue.failed
    ? `${queue.failed} job(s) failed`
    : `${queue.pending || 0} job(s) pending`;
  elements.retry.hidden = !queue.failed;

  const missing = info.missing_files || 0;
  elements.pruneState.textContent = missing
    ? `${missing.toLocaleString()} asset(s) have no file left`
    : "every asset still has a file";
  elements.prune.disabled = missing === 0;

  // One progress line for whichever download is running, because only one ever
  // is - the runner refuses a second.
  if (fetching.running) {
    const share = fetching.total
      ? ` ${Math.floor((fetching.done / fetching.total) * 100)}%`
      : "";
    elements.fetchState.textContent = `${fetching.label || "downloading"}${share}`;
    elements.fetchState.hidden = false;
  } else if (fetching.error) {
    elements.fetchState.textContent = `failed: ${fetching.error}`;
    elements.fetchState.hidden = false;
  } else {
    elements.fetchState.hidden = true;
  }
}

const megabytes = (bytes) =>
  bytes ? `${Math.round(bytes / 1e6).toLocaleString()} MB` : "";

// --- roots ------------------------------------------------------------------

// A typed path rather than a folder picker, and not for want of trying: a
// browser hands back the files inside a chosen directory and never its absolute
// path, which is exactly the one thing the server needs.
async function addRoot() {
  const path = elements.rootPath.value.trim();
  if (!path) return;

  try {
    const result = await api.addRoot(path, {
      vendor: elements.rootVendor.checked,
    });
    elements.rootPath.value = "";
    elements.rootVendor.checked = false;
    await refreshRoots();
    render();
    hooks.onToast?.(`Added ${result.name}`);

    // Offered rather than done, because a first root is often one of several
    // and a scan started per folder walks the earlier ones again for nothing.
    if (window.confirm(`Scan ${result.name} now?`)) {
      await api.startScan({});
      hooks.onToast?.("Scan started");
    }
  } catch (error) {
    hooks.onToast?.(String(error.message || error));
  }
}

async function removeRoot(root) {
  const ok = window.confirm(
    `Stop indexing ${root.path}?\n\n` +
      `The ${root.count.toLocaleString()} asset(s) already indexed are kept, ` +
      `along with their tags and notes. Prune removes the ones whose files are gone.`,
  );
  if (!ok) return;

  try {
    await api.removeRoot(root.path);
    await refreshRoots();
    render();
    hooks.onToast?.(`Removed ${root.name}`);
  } catch (error) {
    hooks.onToast?.(String(error.message || error));
  }
}

// --- extras -----------------------------------------------------------------

async function downloadWeights() {
  try {
    const result = await api.downloadModel();
    hooks.onToast?.(
      result.started
        ? `Downloading ${result.label} (${megabytes(result.bytes)})…`
        : result.reason,
    );
  } catch (error) {
    hooks.onToast?.(String(error.message || error));
  }
}

async function pullVlm() {
  try {
    const result = await api.pullVlm();
    hooks.onToast?.(`Pulling ${result.model} through ollama…`);
  } catch (error) {
    hooks.onToast?.(String(error.message || error));
  }
}

async function runEmbed(redo) {
  if (redo) {
    const ok = window.confirm(
      "Re-embed the whole library?\n\n" +
        "Existing vectors are discarded and rebuilt. Manual tags are untouched.",
    );
    if (!ok) return;
  }

  try {
    const result = await api.startEmbed(redo);
    hooks.onToast?.(
      result.queued
        ? `Embedding ${result.queued.toLocaleString()} asset(s)…`
        : "Everything is already embedded",
    );
    await refreshMaintenance();
    render();
  } catch (error) {
    hooks.onToast?.(String(error.message || error));
  }
}

async function rebuildThumbnails() {
  try {
    const result = await api.startThumbs({ retry: true });
    hooks.onToast?.(
      result.queued
        ? `Rendering ${result.queued.toLocaleString()} thumbnail(s)…`
        : "Every asset that should have a tile has one",
    );
    await refreshMaintenance();
    render();
  } catch (error) {
    hooks.onToast?.(String(error.message || error));
  }
}

async function retryFailed() {
  const result = await api.retryJobs();
  hooks.onToast?.(`Requeued ${result.requeued} job(s)`);
  await refreshMaintenance();
  render();
}

// Two calls: the first reports what would go, and only then is there anything
// worth confirming. Asking before knowing the number is how people click
// through a dialogue that was about to delete four hundred assets.
async function runPrune() {
  try {
    const preview = await api.prune(false);
    if (!preview.count) {
      hooks.onToast?.("Nothing to prune");
      return;
    }

    const names = preview.preview.map((row) => `  ${row.title}`).join("\n");
    const more =
      preview.count > preview.preview.length
        ? `\n  and ${(preview.count - preview.preview.length).toLocaleString()} more`
        : "";
    const ok = window.confirm(
      `Delete ${preview.count.toLocaleString()} asset(s) whose files are gone?\n\n` +
        `${names}${more}\n\n` +
        `Their tags, notes, source and licence go with them. This cannot be undone.`,
    );
    if (!ok) return;

    const result = await api.prune(true);
    hooks.onToast?.(`Deleted ${result.deleted.toLocaleString()} asset(s)`);
    await refreshMaintenance();
    await reload();
    render();
  } catch (error) {
    hooks.onToast?.(String(error.message || error));
  }
}
