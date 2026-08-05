// Wiring: bind the DOM to the store, and own the keyboard map.
//
// The keyboard map is browsing-first, because that is what the layout is tuned
// for. Everything reachable without the mouse:
//
//   /  focus search        s  find similar        r  reveal, or open a link
//   arrows  move           space  quick look      c  copy to destination
//   enter/i  inspector     esc  close, deselect   C  copy path
//   [  toggle sidebar      +/-  thumbnail size    a  select all
//   d  describe with the vision model             ,  settings
//   p  audition audio, and keep playing as the cursor moves

import * as api from "./api.js";
import * as audition from "./audition.js";
import * as grid from "./grid.js";
import * as inspector from "./inspector.js";
import * as quicklook from "./quicklook.js";
import * as query from "./search.js";
import * as settings from "./settings.js";
import {
  state,
  subscribe,
  setQuery,
  setCellSize,
  setCursor,
  select,
  clearSelection,
  selectedAssets,
  refreshSidebar,
  refreshCollections,
  refreshMaintenance,
  reload,
} from "./state.js";

const $ = (id) => document.getElementById(id);

const elements = {
  query: $("query"),
  chips: $("chips"),
  grid: $("grid"),
  canvas: $("canvas"),
  empty: $("empty"),
  emptyText: $("empty-text"),
  emptyAction: $("empty-action"),
  dropzone: $("dropzone"),
  tags: $("tags"),
  kinds: $("kinds"),
  roots: $("roots"),
  collections: $("collections"),
  tagFilter: $("tag-filter"),
  count: $("count"),
  selection: $("selection"),
  progress: $("progress"),
  capabilities: $("capabilities"),
  toast: $("toast"),
};

let tagFilterText = "";

function start() {
  grid.init(elements.grid, elements.canvas, {
    onOpen: (index) => {
      setCursor(index);
      quicklook.open();
    },
  });
  quicklook.init({
    root: $("quicklook"),
    stage: $("ql-stage"),
    title: $("ql-title"),
    meta: $("ql-meta"),
  });
  inspector.init(
    {
      root: $("inspector"),
      close: $("ins-close"),
      heading: $("ins-heading"),
      subheading: $("ins-subheading"),
      preview: $("ins-preview"),
      tags: $("ins-tags"),
      tagInput: $("ins-tag"),
      suggestions: $("ins-suggest"),
      collections: $("ins-collections"),
      collectionPicker: $("ins-collection-add"),
      locations: $("ins-locations"),
      locationsSection: $("ins-locations-section"),
      referenceActions: $("ins-actions"),
      open: $("ins-open"),
      refresh: $("ins-refresh"),
      describe: $("ins-describe"),
      caption: $("ins-caption"),
      "ins-title-field": $("ins-title-field"),
      "ins-source": $("ins-source"),
      "ins-url": $("ins-url"),
      "ins-license": $("ins-license"),
      "ins-caption": $("ins-caption"),
      "ins-notes": $("ins-notes"),
    },
    {
      // A tag or collection clicked inside the panel filters the grid, the same
      // as one clicked in the sidebar. Two ways to reach a filter, one thing
      // that happens.
      onQuery: (term) => setQuery(query.toggleTerm(state.query, ...split(term))),
    },
  );

  settings.init(
    {
      root: $("settings"),
      close: $("set-close"),
      roots: $("set-roots"),
      rootsEmpty: $("set-roots-empty"),
      rootPath: $("set-root-path"),
      rootVendor: $("set-root-vendor"),
      rootAdd: $("set-root-add"),
      clipState: $("set-clip-state"),
      clipDownload: $("set-clip-download"),
      embedState: $("set-embed-state"),
      embed: $("set-embed"),
      embedRedo: $("set-embed-redo"),
      vlmState: $("set-vlm-state"),
      vlmPull: $("set-vlm-pull"),
      thumbsState: $("set-thumbs-state"),
      thumbs: $("set-thumbs"),
      queueState: $("set-queue-state"),
      retry: $("set-retry"),
      pruneState: $("set-prune-state"),
      prune: $("set-prune"),
      fetchState: $("set-fetch"),
    },
    { onToast: toast },
  );

  subscribe(onChange);
  audition.init();
  bindControls();
  bindKeyboard();
  bindImport();
  api.subscribeStatus(onStatus);

  refreshSidebar();
  setQuery(new URLSearchParams(location.search).get("q") || "", {
    immediate: true,
  });
}

// --- rendering --------------------------------------------------------------

function onChange(changed) {
  if (changed.has("results")) {
    grid.reset();
    renderEmpty();
  }
  if (changed.has("layout")) grid.render(true);
  if (changed.has("selection") || changed.has("cursor") || changed.has("playing")) {
    grid.refreshStates();
  }
  if (changed.has("cursor")) {
    grid.scrollToCursor();
    audition.follow(state.assets[state.cursor]);
  }
  if (changed.has("query")) renderQuery();
  if (changed.has("facets")) renderFacets();
  if (changed.has("roots")) {
    renderRoots();
    renderEmpty();
  }
  if (changed.has("collections")) renderCollections();
  if (changed.has("capabilities")) renderCapabilities();
  if (changed.has("roots") || changed.has("maintenance") || changed.has("capabilities")) {
    settings.render();
  }
  if (changed.has("assets")) grid.render(true);
  if (changed.has("selection") || changed.has("cursor") || changed.has("results")) {
    inspector.refresh();
  }
  renderStatus();
}

// "tag:pixel-art" -> ["tag", "pixel-art"], for handing a written filter to the
// same toggle the sidebar uses.
function split(term) {
  const at = term.indexOf(":");
  return [term.slice(0, at), term.slice(at + 1)];
}

function renderQuery() {
  if (document.activeElement !== elements.query) {
    elements.query.value = query.parse(state.query).text.join(" ");
  }

  elements.chips.replaceChildren();
  for (const chip of query.chips(state.query)) {
    const node = document.createElement("span");
    node.className = chip.negated ? "chip negated" : "chip";
    node.append(chip.label);

    const remove = document.createElement("button");
    remove.textContent = "×";
    remove.title = "Remove filter";
    remove.addEventListener("click", () => {
      if (chip.kind === "term") {
        setQuery(query.removeTerm(state.query, chip.index));
      } else if (chip.kind === "sort") {
        setQuery(query.withSort(state.query, query.DEFAULT_SORT));
      } else {
        const parsed = query.parse(state.query);
        parsed.similarTo = null;
        setQuery(query.serialise(parsed));
      }
    });
    node.appendChild(remove);
    elements.chips.appendChild(node);
  }

  const url = new URL(location.href);
  if (state.query) url.searchParams.set("q", state.query);
  else url.searchParams.delete("q");
  history.replaceState(null, "", url);
}

function renderFacets() {
  const needle = tagFilterText.toLowerCase();
  elements.tags.replaceChildren(
    ...state.facets.tags
      .filter((tag) => !needle || tag.name.includes(needle))
      .map((tag) => facetRow("tag", tag.name, tag.count, tag.namespace)),
  );
  elements.kinds.replaceChildren(
    ...state.facets.kinds.map((kind) => facetRow("kind", kind.name, kind.count)),
  );
}

function renderRoots() {
  elements.roots.replaceChildren(
    ...state.roots.map((root) =>
      facetRow("root", root.name, root.count, root.vendor ? "vendor" : null),
    ),
  );
}

function renderCollections() {
  document.getElementById("collections-section").hidden =
    state.collections.length === 0;

  elements.collections.replaceChildren(
    ...state.collections.map((entry) => {
      const row = facetRow("collection", entry.name, entry.count);

      // Export lives on the collection row because that is the noun it acts
      // on. Exporting "the current selection" was the alternative and is a
      // worse fit: the whole point of a collection is that it outlives the
      // selection that built it.
      const send = document.createElement("button");
      send.className = "rowdrop";
      send.textContent = "⤓";
      send.title = `Export ${entry.name} into a project folder`;
      send.addEventListener("click", async (event) => {
        event.stopPropagation();
        await exportCollection(entry);
      });
      row.appendChild(send);

      const drop = document.createElement("button");
      drop.className = "rowdrop";
      drop.textContent = "×";
      drop.title = `Delete the collection ${entry.name}`;
      drop.addEventListener("click", async (event) => {
        event.stopPropagation();
        if (!window.confirm(`Delete the collection "${entry.name}"?`)) return;
        await api.deleteCollection(entry.id);
        await refreshCollections();
        toast(`Deleted ${entry.name}; its assets are untouched`);
      });
      row.appendChild(drop);
      return row;
    }),
  );
}

// Every facet row edits the query and nothing else, which is what keeps the
// sidebar and the query bar two views of one thing rather than two sources of
// truth that disagree.
function facetRow(field, name, count, note = null) {
  const row = document.createElement("li");
  if (
    query.hasTerm(state.query, field, name) ||
    query.hasTerm(state.query, field, name, true)
  ) {
    row.classList.add("active");
  }

  const label = document.createElement("span");
  label.className = "name";
  label.textContent = name;
  label.title = `${field}:${name}`;
  row.appendChild(label);

  if (note) {
    const namespace = document.createElement("span");
    namespace.className = "ns";
    namespace.textContent = note;
    row.appendChild(namespace);
  }

  const total = document.createElement("span");
  total.className = "count";
  total.textContent = count;
  row.appendChild(total);

  row.addEventListener("click", () =>
    setQuery(query.toggleTerm(state.query, field, name)),
  );
  // Right-click excludes rather than includes: the fastest way to say
  // "everything but this" without touching the query bar.
  row.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    setQuery(query.toggleTerm(state.query, field, name, true));
  });
  return row;
}

function renderCapabilities() {
  const missing = [];
  if (!state.capabilities.assimp) missing.push("assimp");
  if (!state.capabilities.ffmpeg) missing.push("ffmpeg");
  elements.capabilities.textContent = missing.length
    ? `no ${missing.join(", ")}`
    : "";
  // The install commands come from the server, because the server is the
  // machine the dependency is missing on and the command differs per platform.
  // One line per tool: on Windows they are not the same line.
  const hints = state.capabilities.install_hints || {};
  elements.capabilities.title = missing.length
    ? missing.map((tool) => hints[tool] || `install ${tool}`).join("\n")
    : "";

  // The search box says what it can do. Whether typing a description works is
  // not something anybody can tell by looking at a text field, and the two ways
  // it can be unavailable need different things done about them - one is a
  // download, the other is a command that has not been run yet.
  const clip = state.capabilities.clip;
  const embedded = state.capabilities.embedded || 0;
  elements.query.placeholder =
    clip && embedded
      ? "Search by words or by description  ( / )"
      : "Search, or type tag:pixel-art w:>=64  ( / )";
  elements.query.title = clip && !embedded
    ? "Semantic search is installed but nothing is embedded: run `assetkeep embed`"
    : "";
}

// An empty library and an empty result set are different problems, and until
// now both said "Nothing matches." The one that mattered was the one where
// nothing *could* match: no folders configured, so Scan walks nothing and
// cheerfully reports four zeroes, with no hint that the thing to fix is one
// panel away. Say which of the three it is, and offer the way out of each.
function renderEmpty() {
  const nothing = state.assets.length === 0 && !state.loading;
  elements.empty.hidden = !nothing;
  if (!nothing) return;

  if (state.query) {
    elements.emptyText.textContent = "Nothing matches.";
    elements.emptyAction.hidden = true;
    return;
  }

  elements.emptyAction.hidden = false;
  if (!state.roots.length) {
    elements.emptyText.textContent = "No folders are being indexed yet.";
    elements.emptyAction.textContent = "Add a folder";
    elements.emptyAction.dataset.action = "settings";
  } else {
    elements.emptyText.textContent = "Nothing indexed yet.";
    elements.emptyAction.textContent = "Scan now";
    elements.emptyAction.dataset.action = "scan";
  }
}

function renderStatus() {
  const shown = state.assets.length;
  elements.count.textContent = state.total
    ? `${shown.toLocaleString()} of ${state.total.toLocaleString()}`
    : state.loading
      ? "searching…"
      : "0";
  elements.selection.textContent = state.selection.size
    ? `${state.selection.size} selected`
    : "";
}

function onStatus(payload) {
  state.status = payload;
  const { scan, queue } = payload;
  const parts = [];
  if (scan.running) parts.push(`scanning ${scan.seen} · ${scan.current || ""}`);
  else if (scan.finished) parts.push(scan.finished);
  // Named by kind, because they cost wildly different amounts of time: forty
  // thumbnails is a moment and forty captions is two minutes.
  const kinds = queue.pending_kinds || {};
  for (const [kind, count] of Object.entries(kinds)) {
    if (count) parts.push(`${count} ${kind}${count === 1 ? "" : "s"} queued`);
  }
  if (queue.failed) parts.push(`${queue.failed} failed`);

  const fetching = payload.fetch || {};
  if (fetching.running) {
    const share = fetching.total
      ? ` ${Math.floor((fetching.done / fetching.total) * 100)}%`
      : "";
    parts.push(`${fetching.label || "downloading"}${share}`);
  }
  elements.progress.textContent = parts.join("  ·  ");

  // A finished download changes what this machine can do, and the capability
  // list is cached for the life of the page - so it is refetched here rather
  // than left insisting the weights are missing until someone reloads.
  const settled = fetching.error || fetching.finished;
  if (!fetching.running && settled && onStatus.lastFetch !== settled) {
    onStatus.lastFetch = settled;
    toast(fetching.error ? `Download failed: ${fetching.error}` : settled);
    api
      .refreshCapabilities()
      .then(refreshSidebar)
      .then(refreshMaintenance)
      .catch(() => {});
  }

  // Redraws the panel, and refetches its counts once the work behind it has
  // settled - otherwise a scan started from the panel finishes without it.
  settings.noteProgress(payload);

  // A caption written by the worker is not visible anywhere until something
  // asks for the asset again, and the panel is where it would be looked for.
  if (!queue.pending) inspector.refreshIfCaptioning();

  // A scan that has just finished has new rows to show, and the queue emptying
  // means thumbnails that were placeholders now exist.
  if (scan.finished && !scan.running && !state.loading) {
    if (onStatus.lastFinished !== scan.finished) {
      onStatus.lastFinished = scan.finished;
      reload().then(refreshSidebar);
    }
  }
}

// --- controls ---------------------------------------------------------------

function bindControls() {
  elements.query.addEventListener("input", () => {
    const parsed = query.parse(state.query);
    parsed.text = elements.query.value.split(/\s+/).filter(Boolean);
    // Typing a filter directly into the box promotes it to a chip, so the two
    // ways of expressing the same thing converge instead of competing.
    const typed = query.parse(elements.query.value);
    if (typed.terms.length) {
      parsed.terms = parsed.terms.concat(typed.terms);
      parsed.text = typed.text;
      if (typed.sort !== query.DEFAULT_SORT) parsed.sort = typed.sort;
      elements.query.value = typed.text.join(" ");
    }
    setQuery(query.serialise(parsed));
  });

  elements.tagFilter.addEventListener("input", (event) => {
    tagFilterText = event.target.value;
    renderFacets();
  });

  $("sidebar-toggle").addEventListener("click", toggleSidebar);
  $("scan-button").addEventListener("click", startScan);
  $("settings-button").addEventListener("click", () => settings.toggle());
  $("add-root").addEventListener("click", (event) => {
    event.stopPropagation();
    settings.open();
  });

  elements.emptyAction.addEventListener("click", () => {
    if (elements.emptyAction.dataset.action === "scan") startScan();
    else settings.open();
  });

  for (const button of document.querySelectorAll("#sizes button")) {
    button.addEventListener("click", () => {
      setCellSize(Number(button.dataset.size));
      for (const other of document.querySelectorAll("#sizes button")) {
        other.classList.toggle("on", other === button);
      }
    });
    if (Number(button.dataset.size) === state.cellSize) button.classList.add("on");
  }

  $("quicklook").addEventListener("click", (event) => {
    if (event.target.id === "quicklook") quicklook.close();
  });

  // Wrapped rather than passed by reference: a listener is called with the
  // click event, and addLink's first parameter is a URL. Without the wrapper
  // the button adds a reference to the string form of a MouseEvent, which is
  // exactly what it did until a headless run caught it.
  $("add-link").addEventListener("click", () => addLink());

  $("new-collection").addEventListener("click", async (event) => {
    event.stopPropagation();
    const assets = selectedAssets();
    const name = window.prompt(
      assets.length
        ? `New collection from ${assets.length} selected asset(s)`
        : "New collection",
    );
    if (!name) return;
    await api.createCollection(name, assets.map((asset) => asset.id));
    await refreshCollections();
    toast(`Created ${name}`);
  });
}

// Scanning with no roots configured is the one case worth intercepting: the
// request succeeds, walks nothing, and reports zeroes, which reads as a broken
// scanner rather than as an empty configuration.
async function startScan() {
  if (!state.roots.length) {
    toast("No folders are being indexed yet");
    settings.open();
    return;
  }
  try {
    await api.startScan({});
    toast("Scan started");
  } catch {
    toast("A scan is already running");
  }
}

// --- references ---------------------------------------------------------------

// Adding a link is deliberately the same gesture as adding a file: the grid
// takes a drop either way, and the sidebar button is there for a URL already on
// the clipboard.
async function addLink(url = null) {
  const target = url || window.prompt("Add a URL");
  if (!target) return;

  toast("Fetching…");
  try {
    const result = await api.addReference(target);
    toast(
      result.created
        ? `Added ${result.title}${result.error ? " (the page did not answer)" : ""}`
        : `Already known: ${result.title}`,
    );
    await reload();
    await refreshSidebar();
  } catch (error) {
    toast(`Could not add that link: ${error.message || error}`);
  }
}

// --- import -----------------------------------------------------------------

// Drag and drop onto the grid, which is the only way into the vault that does
// not involve typing a path. A folder dropped from Finder arrives as a
// directory entry rather than a File, so it has to be walked; without that,
// dropping a pack folder silently imports nothing and looks broken.
//
// A link dragged from another browser tab arrives as text/uri-list rather than
// as a file, and lands here too. One drop target, two kinds of asset, which is
// the whole argument for references being ordinary assets.
function bindImport() {
  let depth = 0;

  const show = (on, links = false) => {
    elements.dropzone.hidden = !on;
    if (!on) return;
    $("drop-title").textContent = links ? "Drop to add a link" : "Drop to import";
    $("drop-hint").textContent = links
      ? "The page is fetched for a title and a preview"
      : "Files, folders and zips go into the vault";
  };

  elements.grid.addEventListener("dragenter", (event) => {
    if (!hasFiles(event) && !hasLinks(event)) return;
    event.preventDefault();
    depth += 1;
    show(true, !hasFiles(event));
  });
  elements.grid.addEventListener("dragover", (event) => {
    if (hasFiles(event) || hasLinks(event)) event.preventDefault();
  });
  elements.grid.addEventListener("dragleave", () => {
    // Counted rather than toggled: dragging over a child element fires leave on
    // the parent, and a naive handler flickers the overlay the whole way across.
    depth = Math.max(0, depth - 1);
    if (!depth) show(false);
  });

  elements.grid.addEventListener("drop", async (event) => {
    if (!hasFiles(event) && !hasLinks(event)) return;
    event.preventDefault();
    depth = 0;
    show(false);

    if (!hasFiles(event)) {
      const dropped = (event.dataTransfer.getData("text/uri-list") || "")
        .split(/\r?\n/)
        .filter((line) => line && !line.startsWith("#"));
      for (const url of dropped) await addLink(url);
      return;
    }

    const files = await collectFiles(event.dataTransfer);
    if (!files.length) return;

    toast(`Importing ${files.length} file(s)…`);
    try {
      const result = await api.importFiles(files);
      const parts = [`${result.imported.length} imported`];
      if (result.duplicates.length) {
        parts.push(`${result.duplicates.length} already known`);
      }
      if (result.skipped.length) parts.push(`${result.skipped.length} skipped`);
      toast(parts.join(", "));
      await reload();
      await refreshSidebar();
    } catch (error) {
      toast(`Import failed: ${error.message || error}`);
    }
  });
}

const hasFiles = (event) =>
  [...(event.dataTransfer?.types || [])].includes("Files");

// A dragged link carries both text/uri-list and text/plain; the first is the
// one that is only ever a URL, so a dragged sentence does not become a
// reference nobody asked for.
const hasLinks = (event) =>
  [...(event.dataTransfer?.types || [])].includes("text/uri-list");

async function collectFiles(transfer) {
  const entries = [...transfer.items]
    .map((item) => item.webkitGetAsEntry?.())
    .filter(Boolean);

  // No entry API - Safari on an older drop, or a synthetic event - so take the
  // flat file list and lose only the folder structure.
  if (!entries.length) return [...transfer.files];

  const files = [];
  for (const entry of entries) await walkEntry(entry, "", files);
  return files;
}

function walkEntry(entry, prefix, out) {
  if (entry.isFile) {
    return new Promise((resolve) => {
      entry.file((file) => {
        // The path is carried in the name, because multipart sends a filename
        // and nothing else. The server splits it back apart, so a dropped
        // folder keeps the structure its tags come from.
        out.push(new File([file], prefix + file.name, { type: file.type }));
        resolve();
      }, resolve);
    });
  }

  const reader = entry.createReader();
  return new Promise((resolve) => {
    const readMore = () => {
      // readEntries returns at most 100 per call and signals the end with an
      // empty batch. Reading once imports the first hundred files of a pack
      // and quietly drops the rest.
      reader.readEntries(async (batch) => {
        if (!batch.length) return resolve();
        for (const child of batch) {
          await walkEntry(child, `${prefix}${entry.name}/`, out);
        }
        readMore();
      }, resolve);
    };
    readMore();
  });
}

// --- keyboard ---------------------------------------------------------------

function bindKeyboard() {
  document.addEventListener("keydown", (event) => {
    const typing =
      event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA";

    if (typing) {
      if (event.key === "Escape") event.target.blur();
      if (event.key === "Enter" && event.target === elements.query) reload();
      return;
    }

    // The settings panel is modal: it owns the keyboard while it is open, so a
    // stray arrow key does not move a cursor nobody can see behind it.
    if (settings.isOpen() && event.key !== "Escape" && event.key !== ",") return;

    const handler = KEYS[event.key];
    if (!handler) return;
    event.preventDefault();
    handler(event);
  });
}

const KEYS = {
  "/": () => elements.query.focus(),
  ArrowRight: () => move(1),
  ArrowLeft: () => move(-1),
  ArrowDown: () => move(grid.columns()),
  ArrowUp: () => move(-grid.columns()),
  " ": () => (quicklook.isOpen() ? quicklook.close() : quicklook.open()),
  Enter: () => inspector.toggle(),
  i: () => inspector.toggle(),
  Escape: () => {
    // One key, unwinding one layer at a time: the modals, then the panel, then
    // the selection. Closing all of them at once is how you lose a selection
    // you spent a minute building because a preview was open over it.
    if (settings.isOpen()) settings.close();
    else if (quicklook.isOpen()) quicklook.close();
    // Before the inspector, because a sound playing is the most recent thing
    // that started and the first thing anyone reaches for Escape to stop.
    else if (audition.isActive()) audition.stop();
    else if (inspector.isOpen()) inspector.close();
    else clearSelection();
  },
  ",": () => settings.toggle(),
  "[": toggleSidebar,
  "+": () => stepSize(1),
  "=": () => stepSize(1),
  "-": () => stepSize(-1),
  s: findSimilar,
  r: revealSelected,
  c: copySelected,
  C: copyPath,
  a: selectAll,
  d: describeSelected,
  p: () => audition.toggle(state.assets[state.cursor]),
};

function move(delta) {
  if (quicklook.isOpen()) {
    quicklook.step(delta > 0 ? Math.sign(delta) : Math.sign(delta));
    return;
  }
  const next = Math.max(0, Math.min(state.assets.length - 1, state.cursor + delta));
  select(next);
}

const SIZES = [96, 128, 192, 256];
function stepSize(direction) {
  const index = SIZES.indexOf(state.cellSize);
  const next = SIZES[Math.max(0, Math.min(SIZES.length - 1, index + direction))];
  setCellSize(next);
  for (const button of document.querySelectorAll("#sizes button")) {
    button.classList.toggle("on", Number(button.dataset.size) === next);
  }
}

function selectAll() {
  for (const asset of state.assets) state.selection.add(asset.id);
  grid.refreshStates();
  renderStatus();
}

function findSimilar() {
  const [asset] = selectedAssets();
  if (!asset) return;
  setQuery(`similar:${asset.id}`, { immediate: true });
}

async function exportCollection(entry) {
  const remembered = state.capabilities.copy_target;
  const destination = window.prompt(
    `Export ${entry.name} (${entry.count} assets) into`,
    remembered || "",
  );
  if (!destination) return;

  toast(`Exporting ${entry.name}…`);
  try {
    const result = await api.exportCollection(entry.id, {
      destination,
      remember: true,
    });
    const parts = [`${result.copied.length} copied`];
    if (result.unchanged.length) parts.push(`${result.unchanged.length} already there`);
    if (result.references.length) parts.push(`${result.references.length} link(s) listed`);
    if (result.missing.length) parts.push(`${result.missing.length} with no file`);
    toast(`${parts.join(", ")} → ${result.destination}`);
    state.capabilities.copy_target = destination;
  } catch (error) {
    toast(String(error.message || error));
  }
}

async function describeSelected() {
  const assets = selectedAssets();
  if (!assets.length) return;
  if (!state.capabilities.vlm) {
    toast(
      state.capabilities.vlm_server
        ? `${state.capabilities.vlm_model} is not installed: assetkeep vlm pull`
        : "ollama is not running",
    );
    return;
  }

  try {
    const result = await api.requestCaptions(assets.map((asset) => asset.id));
    toast(
      result.queued
        ? `Describing ${result.queued} asset(s)…`
        : "Those already have captions",
    );
  } catch (error) {
    toast(String(error.message || error));
  }
}

// Reveal means "show me where this is", and where a reference is is a web
// page. Two behaviours, one key, because it is one intention.
async function revealSelected() {
  const [asset] = selectedAssets();
  if (!asset) return;
  if (asset.kind === "reference") {
    if (asset.source_url) window.open(asset.source_url, "_blank", "noreferrer");
    return;
  }
  try {
    await api.reveal(asset.id);
  } catch {
    toast("Could not reveal that file");
  }
}

async function copySelected() {
  const assets = selectedAssets();
  if (!assets.length) return;

  const remembered = state.capabilities.copy_target;
  const destination = window.prompt("Copy to folder", remembered || "");
  if (!destination) return;

  try {
    const result = await api.copyTo(
      assets.map((asset) => asset.id),
      destination,
    );
    toast(`Copied ${result.copied.length} file(s)`);
    state.capabilities.copy_target = destination;
  } catch (error) {
    toast(String(error.message || error));
  }
}

async function copyPath() {
  const assets = selectedAssets();
  if (!assets.length) return;
  const text = assets.map((asset) => asset.path).filter(Boolean).join("\n");
  try {
    await navigator.clipboard.writeText(text);
    toast(`Copied ${assets.length} path(s)`);
  } catch {
    toast("Clipboard blocked by the browser");
  }
}

function toggleSidebar() {
  document.body.classList.toggle("no-sidebar");
  grid.render(true);
}

let toastTimer = null;
function toast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    elements.toast.hidden = true;
  }, 2200);
}

start();
