// The slide-over: everything about the selection, and the only place metadata
// is edited.
//
// It has two modes and they share one panel. With one asset selected it shows
// that asset; with several it shows what they have in common and edits all of
// them at once. Building the second as a separate screen was the alternative,
// and it is the wrong shape: the selection is the thing being worked on either
// way, and a panel that disappears when you extend the selection to two teaches
// people to only ever select one.
//
// Writes go out on blur rather than on every keystroke, and nothing here waits
// for a reload: an edit patches the row in the store and the grid redraws.

import * as api from "./api.js";
import * as grid from "./grid.js";
import { state, selectedAssets, patchAsset, refreshCollections } from "./state.js";
import { isPixelArt, filename } from "./grid.js";
import { audioFacts } from "./quicklook.js";

//: Fields the panel writes, and the input each one is bound to.
const FIELDS = {
  title: "ins-title-field",
  source_name: "ins-source",
  source_url: "ins-url",
  license: "ins-license",
  caption: "ins-caption",
  notes: "ins-notes",
};

//: How long to wait after a keystroke before asking for tag suggestions. Short
//: enough to feel instantaneous, long enough that typing a whole word is one
//: request rather than nine.
const SUGGEST_DELAY = 90;

//: Tags shown before the list is folded, in bulk mode only. Thirty-five real
//: spritesheets between them carry sixty distinct tags, most on exactly one
//: asset, and rendering all of them pushes every metadata field below the fold
//: to say almost nothing. They are ordered by how many assets carry them, so
//: the fold falls where the information does.
const BULK_TAG_LIMIT = 24;

let root;
let elements = {};
let onQuery = () => {};
let current = null; // the detail payload, or the bulk summary
let bulk = false;
let suggestTimer = null;
let highlighted = -1;
let showAllTags = false;

export function init(nodes, handlers = {}) {
  root = nodes.root;
  elements = nodes;
  onQuery = handlers.onQuery || onQuery;

  elements.close.addEventListener("click", close);
  bindFields();
  bindTagInput();
  bindCollections();
  bindReference();
  bindDescribe();
}

export const isOpen = () => root && !root.hidden;

export function toggle() {
  if (isOpen()) close();
  else open();
}

export function open() {
  if (!selectedAssets().length) return;
  root.hidden = false;
  // The grid's cells are absolutely positioned, so narrowing its container is
  // not enough - the positions have to be recomputed or the rightmost column
  // stays where it was, underneath the panel that just appeared.
  grid.render(true);
  refresh();
}

export function close() {
  root.hidden = true;
  grid.render(true);
  hideSuggestions();
}

// Called whenever the selection or the cursor moves, so the panel follows the
// grid rather than needing to be reopened.
//
// Debounced and generation-guarded, because holding an arrow key moves the
// cursor faster than a round trip completes. Without the first, the panel is a
// request per keypress; without the second, the assets arrive out of order and
// the panel settles on whichever response was slowest rather than on the asset
// actually under the cursor.
let pending = null;
let generation = 0;

export function refresh() {
  clearTimeout(pending);
  pending = setTimeout(refreshNow, 80);
}

async function refreshNow() {
  if (!isOpen()) return;
  const assets = selectedAssets();
  if (!assets.length) {
    close();
    return;
  }

  const mine = ++generation;
  bulk = assets.length > 1;
  showAllTags = false;
  const ids = assets.map((asset) => asset.id);

  const loaded = bulk ? await api.summarise(ids) : await api.getAsset(ids[0]);
  if (mine !== generation) return;

  current = { ...loaded, ids };
  render(assets);
}

function render(assets) {
  elements.heading.textContent = bulk
    ? `${assets.length} assets`
    : filename(assets[0]);
  elements.subheading.textContent = bulk ? summaryLine() : factsLine(assets[0]);

  renderPreview(assets);
  renderTags();
  renderFields();
  renderCollections();
  renderLocations();
  renderReference(assets);
  renderDescribe(assets);
}

function renderPreview(assets) {
  elements.preview.replaceChildren();
  // At most four, because the point is "which assets is this" and a wall of
  // 200 thumbnails answers a question nobody asked.
  for (const asset of assets.slice(0, 4)) {
    const image = document.createElement("img");
    image.src = api.thumbUrl(asset);
    image.alt = "";
    if (isPixelArt(asset)) image.classList.add("pixelated");
    image.addEventListener("error", () => image.remove(), { once: true });
    elements.preview.appendChild(image);
  }
  if (assets.length > 4) {
    const more = document.createElement("span");
    more.className = "more";
    more.textContent = `+${assets.length - 4}`;
    elements.preview.appendChild(more);
  }
}

function summaryLine() {
  return (current.kinds || [])
    .map((kind) => `${kind.count} ${kind.name}`)
    .join("  ·  ");
}

function factsLine(asset) {
  const a = asset.attributes || {};
  const bits = [asset.kind];
  if (a.width && a.height) bits.push(`${a.width | 0} x ${a.height | 0}`);
  if (a.triangles) bits.push(`${(a.triangles | 0).toLocaleString()} tris`);
  if (a.duration) bits.push(`${Number(a.duration).toFixed(2)}s`);
  // Sample rate, channels, depth and level. Probed since M1 and shown nowhere
  // until M6, which made "is this one stereo" a question you could only answer
  // by leaving the tool.
  if (asset.kind === "audio") bits.push(...audioFacts(a));
  if (a.bytes) bits.push(formatBytes(a.bytes));
  return bits.join("  ·  ");
}

// --- tags -------------------------------------------------------------------

function renderTags() {
  const all = bulk ? current.tags : dedupe(current.tags);
  const folded = bulk && !showAllTags && all.length > BULK_TAG_LIMIT;
  const rows = folded ? all.slice(0, BULK_TAG_LIMIT) : all;
  elements.tags.replaceChildren();

  for (const tag of rows) {
    const chip = document.createElement("span");
    chip.className = tag.manual ? "tagchip manual" : "tagchip";
    // In bulk mode a tag that only some of the selection carries is dimmed and
    // labelled, because "add" and "remove" mean different things depending on
    // which it is, and guessing wrong is a bulk edit nobody asked for.
    if (bulk && tag.count < current.count) chip.classList.add("partial");

    const label = document.createElement("button");
    label.className = "tagname";
    label.textContent = tag.name;
    label.title = `Filter by tag:${tag.name}`;
    label.addEventListener("click", () => onQuery(`tag:${tag.name}`));
    chip.appendChild(label);

    if (bulk && tag.count < current.count) {
      const count = document.createElement("span");
      count.className = "tagcount";
      count.textContent = tag.count;
      chip.appendChild(count);
    }

    // Only manual tags get a remove button. An automated one would grow back
    // on the next rescan, so the button would be a lie.
    if (tag.manual || tag.source === "manual") {
      const drop = document.createElement("button");
      drop.className = "tagdrop";
      drop.textContent = "×";
      drop.title = "Remove this tag";
      drop.addEventListener("click", () => removeTag(tag.name));
      chip.appendChild(drop);
    }

    elements.tags.appendChild(chip);
  }

  if (folded) {
    const more = document.createElement("button");
    more.className = "morechips";
    more.textContent = `+${all.length - BULK_TAG_LIMIT} more`;
    more.addEventListener("click", () => {
      showAllTags = true;
      renderTags();
    });
    elements.tags.appendChild(more);
  }
}

// One asset's tags arrive one row per source, so a tag both the tagger and a
// person applied appears twice. Collapse them, keeping manual if either was.
function dedupe(tags) {
  const seen = new Map();
  for (const tag of tags || []) {
    const existing = seen.get(tag.name);
    if (existing) existing.manual = existing.manual || tag.source === "manual";
    else seen.set(tag.name, { ...tag, manual: tag.source === "manual" });
  }
  return [...seen.values()];
}

async function addTag(name) {
  const value = name.trim();
  if (!value) return;
  await api.bulkTags(current.ids, [value], []);
  elements.tagInput.value = "";
  hideSuggestions();
  await refreshNow();
}

async function removeTag(name) {
  await api.bulkTags(current.ids, [], [name]);
  await refreshNow();
}

function bindTagInput() {
  elements.tagInput.addEventListener("input", () => {
    clearTimeout(suggestTimer);
    suggestTimer = setTimeout(suggest, SUGGEST_DELAY);
  });

  elements.tagInput.addEventListener("keydown", (event) => {
    const options = [...elements.suggestions.children];

    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const step = event.key === "ArrowDown" ? 1 : -1;
      highlight(Math.max(-1, Math.min(options.length - 1, highlighted + step)));
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      // The highlighted suggestion if there is one, otherwise exactly what was
      // typed. Committing the first suggestion on a bare Enter is the classic
      // way an autocomplete applies a tag nobody chose.
      const chosen = options[highlighted];
      addTag(chosen ? chosen.dataset.name : elements.tagInput.value);
      return;
    }
    if (event.key === "Escape") {
      event.stopPropagation();
      if (options.length) hideSuggestions();
      else elements.tagInput.blur();
    }
  });

  elements.tagInput.addEventListener("blur", () => {
    // Deferred, or clicking a suggestion closes the list before the click
    // registers on it.
    setTimeout(hideSuggestions, 120);
  });
}

async function suggest() {
  const prefix = elements.tagInput.value.trim();
  if (!prefix) {
    hideSuggestions();
    return;
  }

  const { tags } = await api.suggestTags(prefix);
  const already = new Set((current.tags || []).map((tag) => tag.name));
  const rows = tags.filter((tag) => !already.has(tag.name)).slice(0, 8);
  if (!rows.length) {
    hideSuggestions();
    return;
  }

  elements.suggestions.replaceChildren(
    ...rows.map((tag) => {
      const item = document.createElement("li");
      item.dataset.name = tag.name;
      item.innerHTML = `<span></span><em></em>`;
      item.querySelector("span").textContent = tag.name;
      item.querySelector("em").textContent = tag.count;
      item.addEventListener("mousedown", (event) => {
        event.preventDefault();
        addTag(tag.name);
      });
      return item;
    }),
  );
  elements.suggestions.hidden = false;
  highlight(-1);
}

function highlight(index) {
  highlighted = index;
  [...elements.suggestions.children].forEach((item, at) => {
    item.classList.toggle("on", at === index);
  });
}

function hideSuggestions() {
  elements.suggestions.hidden = true;
  elements.suggestions.replaceChildren();
  highlighted = -1;
}

// --- metadata ---------------------------------------------------------------

function renderFields() {
  const source = bulk ? current.fields || {} : current;
  for (const [field, id] of Object.entries(FIELDS)) {
    const input = elements[id];
    // Title is per-asset by definition; offering to set forty titles to one
    // string is offering to destroy them.
    input.closest(".field").hidden = bulk && field === "title";
    input.value = source[field] || "";
    input.placeholder =
      bulk && source[field] === null ? "mixed - type to set all" : "";
  }
}

function bindFields() {
  for (const [field, id] of Object.entries(FIELDS)) {
    const input = elements[id];
    input.addEventListener("change", () => save(field, input.value));
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && input.tagName !== "TEXTAREA") input.blur();
      if (event.key === "Escape") {
        // Leave the field, keep the panel. Escape unwinds one layer, and from
        // inside a half-typed licence the layer being left is the field.
        event.stopPropagation();
        input.blur();
      }
    });
  }
}

async function save(field, value) {
  if (!current) return;
  if (bulk) {
    await api.bulkEdit(current.ids, { [field]: value });
    for (const id of current.ids) patchAsset(id, { [field]: value });
  } else {
    const updated = await api.editAsset(current.ids[0], { [field]: value });
    patchAsset(current.ids[0], { [field]: value, title: updated.title });
    current = { ...updated, ids: current.ids };
  }
}

// --- collections ------------------------------------------------------------

function renderCollections() {
  elements.collections.replaceChildren();
  const memberships = bulk ? [] : current.collections || [];

  for (const entry of memberships) {
    const chip = document.createElement("span");
    chip.className = "tagchip manual";

    const label = document.createElement("button");
    label.className = "tagname";
    label.textContent = entry.name;
    label.addEventListener("click", () => onQuery(`collection:${entry.name}`));
    chip.appendChild(label);

    const drop = document.createElement("button");
    drop.className = "tagdrop";
    drop.textContent = "×";
    drop.title = "Take out of this collection";
    drop.addEventListener("click", async () => {
      await api.removeFromCollection(entry.id, current.ids);
      await Promise.all([refreshNow(), refreshCollections()]);
    });
    chip.appendChild(drop);
    elements.collections.appendChild(chip);
  }

  elements.collectionPicker.replaceChildren(
    new Option(bulk ? `Add ${current.count} to…` : "Add to…", ""),
    ...state.collections.map(
      (entry) => new Option(`${entry.name}  (${entry.count})`, entry.id),
    ),
    new Option("New collection…", "new"),
  );
}

function bindCollections() {
  elements.collectionPicker.addEventListener("change", async (event) => {
    const value = event.target.value;
    event.target.value = "";
    if (!value || !current) return;

    if (value === "new") {
      const name = window.prompt("New collection");
      if (!name) return;
      await api.createCollection(name, current.ids);
    } else {
      await api.addToCollection(Number(value), current.ids);
    }
    await Promise.all([refreshNow(), refreshCollections()]);
  });
}

// --- references -------------------------------------------------------------

// A reference has one action a file does not - go to the page - and one the
// file version of would be meaningless: fetch it again. Both are hidden for
// everything else rather than shown disabled, the same rule the whole UI
// follows for capabilities.
function renderReference(assets) {
  const isReference = !bulk && assets[0].kind === "reference";
  elements.referenceActions.hidden = !isReference;
  if (!isReference) return;

  elements.open.href = current.source_url || "#";
  elements.open.textContent = current.source_url
    ? `Open ${hostOf(current.source_url)}`
    : "No URL";
}

function hostOf(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function bindReference() {
  elements.refresh.addEventListener("click", async () => {
    if (!current) return;
    elements.refresh.disabled = true;
    try {
      const result = await api.refreshReference(current.ids[0]);
      patchAsset(current.ids[0], {
        title: result.asset.title,
        // The tile is content-addressed by a hash that did not change, so a
        // newly fetched preview would be served from cache without this.
        ...(result.thumbnail ? { thumb_version: Date.now() } : {}),
      });
      await refreshNow();
      grid.render(true);
    } finally {
      elements.refresh.disabled = false;
    }
  });
}

// --- captions ---------------------------------------------------------------

function renderDescribe(assets) {
  // Hidden without a vision model, and hidden for kinds it cannot see. A
  // waveform described as a picture produces a confident and entirely fictional
  // account of what the sound is of.
  const captionable = assets.every(
    (asset) => asset.kind === "image" || asset.kind === "model3d",
  );
  elements.describe.hidden = !state.capabilities.vlm || !captionable;
  elements.describe.textContent = bulk ? `Describe ${assets.length}` : "Describe";
  // Bulk mode shares one caption box between assets that each want their own,
  // so the field is only editable one at a time. The button still works on the
  // whole selection.
  elements.caption.disabled = bulk;
  elements.caption.placeholder = bulk ? "one per asset" : "";
}

function bindDescribe() {
  elements.describe.addEventListener("click", async () => {
    if (!current) return;
    elements.describe.disabled = true;
    try {
      const result = await api.requestCaptions(current.ids);
      elements.caption.placeholder =
        result.queued > 0 ? "describing…" : "already described";
    } catch (error) {
      elements.caption.placeholder = String(error.message || error).slice(0, 80);
    } finally {
      elements.describe.disabled = false;
    }
  });
}

// Called when the job queue goes idle, so a caption that has just been written
// appears without anybody reopening the panel.
export function refreshIfCaptioning() {
  if (isOpen() && elements.caption.placeholder === "describing…") refreshNow();
}

// --- locations --------------------------------------------------------------

function renderLocations() {
  elements.locations.replaceChildren();
  // A reference has no location and never will; the section is not empty, it
  // is inapplicable.
  const isReference = !bulk && (selectedAssets()[0] || {}).kind === "reference";
  elements.locationsSection.hidden = bulk || isReference;
  if (bulk || isReference) return;

  for (const location of current.locations || []) {
    const row = document.createElement("li");
    row.textContent = location.abs_path;
    row.title = location.abs_path;
    if (!location.present) row.classList.add("absent");
    elements.locations.appendChild(row);
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes | 0} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
