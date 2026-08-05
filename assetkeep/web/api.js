// Thin fetch wrappers. The capability list is fetched once and cached, because
// every part of the UI asks it what to hide and it cannot change while the page
// is open.

let capabilities = null;

async function request(path, options = {}) {
  // Only a string body is JSON. FormData has to be left alone: setting the
  // header ourselves omits the multipart boundary the browser was about to
  // generate, and the server then rejects a body it cannot split.
  const json = typeof options.body === "string";
  const response = await fetch(path, {
    headers: json ? { "Content-Type": "application/json" } : {},
    ...options,
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => response.statusText);
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.status === 204 ? null : response.json();
}

const post = (path, body) =>
  request(path, { method: "POST", body: JSON.stringify(body ?? {}) });

const patch = (path, body) =>
  request(path, { method: "PATCH", body: JSON.stringify(body ?? {}) });

const remove = (path, body) =>
  request(path, {
    method: "DELETE",
    ...(body ? { body: JSON.stringify(body) } : {}),
  });

export async function getCapabilities() {
  if (capabilities === null) capabilities = await request("/api/capabilities");
  return capabilities;
}

export const searchAssets = (query, limit, offset) =>
  request(
    `/api/assets?q=${encodeURIComponent(query)}&limit=${limit}&offset=${offset}`,
  );

export const countAssets = (query) =>
  request(`/api/assets/count?q=${encodeURIComponent(query)}`);

export const getAsset = (id) => request(`/api/assets/${id}`);

export const getSimilar = (id) => request(`/api/assets/${id}/similar`);

export const getFacets = (query) =>
  request(`/api/facets?q=${encodeURIComponent(query)}`);

export const getRoots = () => request("/api/roots");

export const startScan = (body) => post("/api/scan", body);

export const reveal = (id) => post(`/api/assets/${id}/reveal`);

export const copyTo = (ids, destination, remember = true) =>
  post("/api/assets/copy", { ids, destination, remember });

export const addTags = (id, tags) => post(`/api/assets/${id}/tags`, { tags });

export const removeTag = (id, tag) =>
  remove(`/api/assets/${id}/tags/${encodeURIComponent(tag)}`);

export const editAsset = (id, fields) => patch(`/api/assets/${id}`, fields);

export const summarise = (ids) => post("/api/assets/summary", { ids });

export const bulkTags = (ids, add = [], drop = []) =>
  post("/api/assets/bulk/tags", { ids, add, remove: drop });

export const bulkEdit = (ids, fields) => patch("/api/assets/bulk", { ids, ...fields });

export const suggestTags = (prefix) =>
  request(`/api/tags?prefix=${encodeURIComponent(prefix)}`);

export const getCollections = () => request("/api/collections");

export const createCollection = (name, ids = []) =>
  post("/api/collections", { name, ids });

export const renameCollection = (id, name) =>
  patch(`/api/collections/${id}`, { name });

export const deleteCollection = (id) => remove(`/api/collections/${id}`);

export const addToCollection = (id, ids) =>
  post(`/api/collections/${id}/assets`, { ids });

export const removeFromCollection = (id, ids) =>
  remove(`/api/collections/${id}/assets`, { ids });

// Multipart rather than JSON, so the browser streams the file instead of
// base64-ing a 200 MB pack into a string first.
export function importFiles(files, { batch = "", collection = "" } = {}) {
  const form = new FormData();
  for (const file of files) form.append("files", file, file.name);
  if (batch) form.append("batch", batch);
  if (collection) form.append("collection", collection);
  return request("/api/import", { method: "POST", body: form });
}

// A URL is added synchronously and appears in the grid on the next reload; the
// server fetches the page and its preview inside this one request.
export const addReference = (url, options = {}) =>
  post("/api/references", { url, ...options });

export const refreshReference = (id, overwrite = false) =>
  post(`/api/references/${id}/refresh`, { overwrite });

export const exportCollection = (id, body) =>
  post(`/api/collections/${id}/export`, body);

// Captions are queued, not awaited: a selection of forty is minutes of model
// time, and the status stream already reports the queue draining.
export const requestCaptions = (ids, redo = false) =>
  post("/api/captions", { ids, redo });

export const retryJobs = () => post("/api/jobs/retry");

// The version suffix is only ever set by something that has just replaced a
// tile in place - refetching a reference's preview - because the URL is
// content-addressed and would otherwise be served from cache forever, which is
// exactly the behaviour wanted everywhere else.
export const thumbUrl = (asset) =>
  asset.thumb_version
    ? `/api/thumb/${asset.content_hash}?v=${asset.thumb_version}`
    : `/api/thumb/${asset.content_hash}`;
export const fileUrl = (asset) => `/api/file/${asset.id}`;

// Server-sent events for scan and queue progress. Reconnects on its own, which
// matters because the browser drops the stream whenever the laptop sleeps.
export function subscribeStatus(onMessage) {
  const source = new EventSource("/api/scan/status");
  source.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data));
    } catch {
      /* heartbeats carry no data */
    }
  };
  return source;
}
