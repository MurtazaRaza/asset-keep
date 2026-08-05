// One store. The query string is the canonical state of what is displayed.
//
// Everything that filters - typing in the bar, clicking a tag, clicking a root,
// asking for similar - edits the query and nothing else. Without that rule the
// sidebar and the bar end up as two sources of truth that disagree about what
// is on screen, which is the usual way this kind of UI rots.

import * as api from "./api.js";

const PAGE = 200;

export const state = {
  query: "",
  assets: [],
  total: 0,
  loading: false,
  exhausted: false,
  selection: new Set(),
  cursor: -1,
  cellSize: 128,
  facets: { tags: [], kinds: [] },
  roots: [],
  collections: [],
  capabilities: {},
  // Counts that move while the page is open - embedded, captioned, outstanding
  // - as opposed to `capabilities`, which cannot and is cached for the session.
  maintenance: {},
  status: { scan: {}, queue: {}, fetch: {} },
};

const listeners = new Set();

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function emit(...what) {
  for (const listener of listeners) listener(new Set(what));
}

// A generation counter, so a slow response for an old query cannot overwrite
// the results of a newer one. Typing quickly makes this happen constantly.
let generation = 0;

export async function setQuery(query, { immediate = false } = {}) {
  state.query = query;
  emit("query");
  if (immediate) await reload();
  else scheduleReload();
}

let reloadTimer = null;
function scheduleReload() {
  clearTimeout(reloadTimer);
  reloadTimer = setTimeout(reload, 160);
}

export async function reload() {
  const mine = ++generation;
  state.loading = true;
  state.exhausted = false;
  emit("loading");

  try {
    const [page, count, facets] = await Promise.all([
      api.searchAssets(state.query, PAGE, 0),
      api.countAssets(state.query),
      api.getFacets(state.query),
    ]);
    if (mine !== generation) return;

    state.assets = page.assets;
    state.total = count.count;
    state.facets = facets;
    state.exhausted = page.assets.length >= count.count;
    state.selection.clear();
    state.cursor = page.assets.length ? 0 : -1;
  } finally {
    if (mine === generation) {
      state.loading = false;
      emit("results", "facets", "selection");
    }
  }
}

export async function loadMore() {
  if (state.loading || state.exhausted) return;
  const mine = generation;
  state.loading = true;

  try {
    const page = await api.searchAssets(state.query, PAGE, state.assets.length);
    if (mine !== generation) return;
    if (!page.assets.length) {
      state.exhausted = true;
      return;
    }
    state.assets = state.assets.concat(page.assets);
    state.exhausted = state.assets.length >= state.total;
    emit("results");
  } finally {
    if (mine === generation) state.loading = false;
  }
}

export function setCellSize(size) {
  state.cellSize = size;
  emit("layout");
}

export function setCursor(index) {
  if (!state.assets.length) return;
  state.cursor = Math.max(0, Math.min(state.assets.length - 1, index));
  emit("cursor");
}

export function select(index, { toggle = false, range = false } = {}) {
  const asset = state.assets[index];
  if (!asset) return;

  if (range && state.cursor >= 0) {
    const [from, to] = [state.cursor, index].sort((a, b) => a - b);
    for (let i = from; i <= to; i += 1) state.selection.add(state.assets[i].id);
  } else if (toggle) {
    if (state.selection.has(asset.id)) state.selection.delete(asset.id);
    else state.selection.add(asset.id);
  } else {
    state.selection.clear();
    state.selection.add(asset.id);
  }

  state.cursor = index;
  emit("selection", "cursor");
}

export function clearSelection() {
  state.selection.clear();
  emit("selection");
}

export function selectedAssets() {
  if (state.selection.size) {
    return state.assets.filter((asset) => state.selection.has(asset.id));
  }
  const current = state.assets[state.cursor];
  return current ? [current] : [];
}

export async function refreshSidebar() {
  const [roots, collections, capabilities] = await Promise.all([
    api.getRoots(),
    api.getCollections(),
    api.getCapabilities(),
  ]);
  state.roots = roots.roots;
  state.collections = collections.collections;
  state.capabilities = capabilities;
  emit("roots", "collections", "capabilities");
}

export async function refreshCollections() {
  state.collections = (await api.getCollections()).collections;
  emit("collections");
}

export async function refreshRoots() {
  state.roots = (await api.getRoots()).roots;
  emit("roots");
}

export async function refreshMaintenance() {
  state.maintenance = await api.getMaintenance();
  emit("maintenance");
}

// Fold an edited asset back into the loaded page without refetching it.
//
// Re-running the search instead would be simpler and is wrong: an edit can
// change whether the asset still matches the query - tagging something `wip`
// while looking at `-tag:wip` - and having it vanish out from under the panel
// mid-edit is not a behaviour anyone wants from a metadata field.
export function patchAsset(id, fields) {
  const asset = state.assets.find((candidate) => candidate.id === id);
  if (!asset) return;
  Object.assign(asset, fields);
  emit("assets");
}
