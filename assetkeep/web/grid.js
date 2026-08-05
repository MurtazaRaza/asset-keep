// Virtualised grid: only the visible window plus a couple of rows exists in the
// DOM, so a 50,000 item library scrolls at full speed.
//
// Fixed square cells rather than a justified masonry layout. With mixed aspect
// ratios uniform cells scan far better, and sprites - which are what this is
// for - are mostly square or close to it anyway.

import { state, select, setCursor, loadMore } from "./state.js";
import * as api from "./api.js";
import * as audition from "./audition.js";

const GAP = 8;
const PADDING = 12;
const BUFFER_ROWS = 2;

let container;
let canvas;
let onOpen = () => {};
const mounted = new Map(); // index -> element

export function init(gridElement, canvasElement, handlers = {}) {
  container = gridElement;
  canvas = canvasElement;
  onOpen = handlers.onOpen || onOpen;

  container.addEventListener("scroll", () => {
    render();
    // Fetch the next page well before the bottom, so scrolling never stalls on
    // a network round trip.
    const remaining =
      container.scrollHeight - container.scrollTop - container.clientHeight;
    if (remaining < container.clientHeight) loadMore();
  });
  window.addEventListener("resize", () => render(true));
}

export function columns() {
  const usable = container.clientWidth - PADDING * 2 + GAP;
  return Math.max(1, Math.floor(usable / (state.cellSize + GAP)));
}

export function render(force = false) {
  if (!container) return;

  const perRow = columns();
  const step = state.cellSize + GAP;
  const rows = Math.ceil(state.assets.length / perRow);
  canvas.style.height = `${rows * step + PADDING * 2}px`;

  if (force) {
    for (const element of mounted.values()) element.remove();
    mounted.clear();
  }

  const firstRow = Math.max(
    0,
    Math.floor((container.scrollTop - PADDING) / step) - BUFFER_ROWS,
  );
  const lastRow = Math.min(
    rows - 1,
    Math.ceil((container.scrollTop + container.clientHeight) / step) + BUFFER_ROWS,
  );

  const wanted = new Set();
  for (let row = firstRow; row <= lastRow; row += 1) {
    for (let column = 0; column < perRow; column += 1) {
      const index = row * perRow + column;
      if (index >= state.assets.length) break;
      wanted.add(index);

      let element = mounted.get(index);
      if (!element) {
        element = build(index);
        mounted.set(index, element);
        canvas.appendChild(element);
      }
      element.style.transform =
        `translate(${PADDING + column * step}px, ${PADDING + row * step}px)`;
      element.style.width = `${state.cellSize}px`;
      element.style.height = `${state.cellSize}px`;
      applyStateClasses(element, index);
    }
  }

  for (const [index, element] of mounted) {
    if (!wanted.has(index)) {
      element.remove();
      mounted.delete(index);
    }
  }
}

// Rebuild everything: the asset at a given index has changed identity.
export function reset() {
  container.scrollTop = 0;
  render(true);
}

export function refreshStates() {
  for (const [index, element] of mounted) applyStateClasses(element, index);
}

export function scrollToCursor() {
  const perRow = columns();
  const step = state.cellSize + GAP;
  const row = Math.floor(state.cursor / perRow);
  const top = PADDING + row * step;
  const bottom = top + state.cellSize;

  if (top < container.scrollTop) {
    container.scrollTop = top - PADDING;
  } else if (bottom > container.scrollTop + container.clientHeight) {
    container.scrollTop = bottom - container.clientHeight + PADDING;
  }
  render();
}

function build(index) {
  const asset = state.assets[index];
  const element = document.createElement("div");
  element.className = "cell";
  element.draggable = true;

  const image = document.createElement("img");
  image.loading = "lazy";
  image.decoding = "async";
  image.alt = asset.title;
  image.src = api.thumbUrl(asset);
  if (isPixelArt(asset)) image.classList.add("pixelated");
  // A model with no assimp, or an Aseprite file, has no thumbnail and never
  // will. Showing a broken-image glyph is worse than showing the kind badge.
  image.addEventListener("error", () => image.remove(), { once: true });
  element.appendChild(image);

  if (asset.kind !== "image" || !asset.present) {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = asset.present ? kindLabel(asset) : "missing";
    element.appendChild(badge);
  }

  // A play button on the tile itself, for the same reason `p` exists: the
  // waveform is the only thumbnail in the grid that is a picture of something
  // you are meant to hear, and clicking it is the obvious thing to try.
  if (asset.kind === "audio" && asset.present) {
    const play = document.createElement("button");
    play.className = "playbutton";
    play.title = "Audition (p)";
    play.textContent = "▶";
    play.addEventListener("mousedown", (event) => event.stopPropagation());
    play.addEventListener("click", (event) => {
      event.stopPropagation();
      setCursor(index);
      audition.toggle(asset);
    });
    element.appendChild(play);
  }

  const label = document.createElement("div");
  label.className = "label";
  label.innerHTML = `<b></b><span></span>`;
  label.querySelector("b").textContent = asset.title;
  label.querySelector("span").textContent = shape(asset);
  element.appendChild(label);

  element.addEventListener("mousedown", (event) => {
    select(index, { toggle: event.metaKey || event.ctrlKey, range: event.shiftKey });
  });
  element.addEventListener("dblclick", () => onOpen(index));

  element.addEventListener("dragstart", (event) => {
    setCursor(index);
    // A reference has no bytes to hand over; what it drags out is its URL,
    // which is what a browser or an editor will accept anyway.
    if (asset.kind === "reference") {
      event.dataTransfer.setData("text/uri-list", asset.source_url || "");
      event.dataTransfer.setData("text/plain", asset.source_url || asset.title);
      return;
    }
    // Chromium honours DownloadURL and writes the real file to the drop target,
    // which is the only way a browser can hand a file to Finder. Everything
    // else ignores it, so this is a bonus on top of Reveal and Copy-to rather
    // than the primary way out.
    const url = new URL(api.fileUrl(asset), location.href).href;
    event.dataTransfer.setData(
      "DownloadURL",
      `application/octet-stream:${filename(asset)}:${url}`,
    );
    event.dataTransfer.setData("text/plain", asset.path || asset.title);
  });

  return element;
}

function applyStateClasses(element, index) {
  const asset = state.assets[index];
  element.classList.toggle("selected", state.selection.has(asset.id));
  element.classList.toggle("cursor", state.cursor === index);
  element.classList.toggle("missing", !asset.present);
  element.classList.toggle("playing", audition.playingId() === asset.id);
}

export function isPixelArt(asset) {
  const { block_size: block, width } = asset.attributes || {};
  // Either it was measured as an upscaled grid, or it is small enough that the
  // grid will be upscaling it now.
  return Boolean(block) || (width && width <= state.cellSize);
}

function kindLabel(asset) {
  return { model3d: "3D", audio: "audio", reference: "link" }[asset.kind] || asset.kind;
}

function shape(asset) {
  const a = asset.attributes || {};
  if (asset.kind === "reference") return a.host || "";
  if (a.width && a.height) return `${a.width | 0}x${a.height | 0}`;
  if (a.triangles) return `${a.triangles | 0} tris`;
  if (a.duration) return `${Number(a.duration).toFixed(1)}s`;
  return "";
}

export function filename(asset) {
  return asset.path ? asset.path.split("/").pop() : asset.title;
}
