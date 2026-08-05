// Fullscreen preview. Arrows keep moving through results while it is open,
// which is the whole point: it is a way to look through a result set, not a
// modal you enter and leave for each file.

import { state, setCursor } from "./state.js";
import * as api from "./api.js";
import { isPixelArt, filename } from "./grid.js";

const FRAME_MS = 120;
const MAX_SHEET_ZOOM = 12;

let root;
let stage;
let titleEl;
let metaEl;
let timer = null;

export function init(elements) {
  root = elements.root;
  stage = elements.stage;
  titleEl = elements.title;
  metaEl = elements.meta;
}

export const isOpen = () => root && !root.hidden;

export function open() {
  if (!state.assets[state.cursor]) return;
  root.hidden = false;
  show();
}

export function close() {
  stopAnimation();
  root.hidden = true;
  stage.replaceChildren();
}

export function step(delta) {
  setCursor(state.cursor + delta);
  if (isOpen()) show();
}

export function show() {
  const asset = state.assets[state.cursor];
  if (!asset) return;

  stopAnimation();
  stage.replaceChildren();
  titleEl.textContent = filename(asset);
  metaEl.textContent = describe(asset);

  if (asset.kind === "audio") stage.appendChild(audioPlayer(asset));
  else if (asset.kind === "reference") stage.appendChild(linkCard(asset));
  else if (isSheet(asset)) stage.appendChild(sheetPlayer(asset));
  else stage.appendChild(stillImage(asset));
}

// A reference has nothing to enlarge - its preview is already the whole of what
// was fetched - so the fullscreen view is where the link itself goes, at a size
// that can be read and clicked.
function linkCard(asset) {
  const stack = document.createElement("div");
  stack.className = "stack";

  const preview = document.createElement("img");
  preview.src = api.thumbUrl(asset);
  preview.alt = "";
  preview.addEventListener("error", () => preview.remove(), { once: true });

  const link = document.createElement("a");
  link.className = "linkout";
  link.href = asset.source_url || "#";
  link.target = "_blank";
  link.rel = "noreferrer noopener";
  link.textContent = asset.source_url || "no URL recorded";

  stack.append(preview, link);
  if (asset.notes) {
    const notes = document.createElement("p");
    notes.className = "linknotes";
    notes.textContent = asset.notes;
    stack.appendChild(notes);
  }
  return stack;
}

function stillImage(asset) {
  const image = document.createElement("img");
  // The original rather than the thumbnail: at fullscreen a 256px tile is
  // exactly the wrong thing to show, and this is also the only place a
  // 4096px texture can actually be looked at.
  image.src = asset.kind === "image" ? api.fileUrl(asset) : api.thumbUrl(asset);
  image.alt = asset.title;
  if (isPixelArt(asset)) image.classList.add("pixelated");
  image.addEventListener(
    "error",
    () => {
      image.src = api.thumbUrl(asset);
    },
    { once: true },
  );
  return image;
}

function audioPlayer(asset) {
  const stack = document.createElement("div");
  stack.className = "stack";

  const waveform = document.createElement("img");
  waveform.src = api.thumbUrl(asset);
  waveform.alt = "";
  waveform.addEventListener("error", () => waveform.remove(), { once: true });

  const player = document.createElement("audio");
  player.controls = true;
  player.autoplay = true;
  player.src = api.fileUrl(asset);

  stack.append(waveform, player);
  return stack;
}

// A detected spritesheet plays its frames. Showing the sheet is showing what
// the file looks like; playing it is showing what it is, and whether the walk
// cycle is any good is the actual question being asked of it.
function sheetPlayer(asset) {
  const { cols, rows, width, height } = asset.attributes;
  const frameWidth = width / cols;
  const frameHeight = height / rows;

  // Whole-number zoom only, so a 16px frame stays a grid of crisp squares
  // rather than a blurred or unevenly-sampled one. Capped so that a 4px
  // particle does not fill the screen.
  const available = Math.min(window.innerWidth * 0.5, window.innerHeight * 0.6);
  const zoom = Math.min(
    MAX_SHEET_ZOOM,
    Math.max(1, Math.floor(available / Math.max(frameWidth, frameHeight))),
  );

  const view = document.createElement("div");
  view.className = "sheet";
  view.style.width = `${frameWidth * zoom}px`;
  view.style.height = `${frameHeight * zoom}px`;
  view.style.backgroundImage = `url(${api.fileUrl(asset)})`;
  view.style.backgroundSize = `${width * zoom}px ${height * zoom}px`;

  let frame = 0;
  const total = cols * rows;
  const advance = () => {
    const column = frame % cols;
    const row = Math.floor(frame / cols) % rows;
    view.style.backgroundPosition =
      `-${column * frameWidth * zoom}px -${row * frameHeight * zoom}px`;
    frame = (frame + 1) % total;
  };

  advance();
  timer = setInterval(advance, FRAME_MS);
  return view;
}

function stopAnimation() {
  if (timer !== null) {
    clearInterval(timer);
    timer = null;
  }
}

function isSheet(asset) {
  const a = asset.attributes || {};
  return asset.kind === "image" && a.cols > 0 && a.rows > 0 && a.cols * a.rows > 1;
}

function describe(asset) {
  const a = asset.attributes || {};
  const bits = [];
  if (a.width && a.height) bits.push(`${a.width | 0} x ${a.height | 0}`);
  if (a.cols && a.rows) bits.push(`${a.cols | 0}x${a.rows | 0} sheet`);
  if (a.triangles) bits.push(`${(a.triangles | 0).toLocaleString()} tris`);
  if (a.duration) bits.push(`${Number(a.duration).toFixed(2)}s`);
  if (a.bytes) bits.push(formatBytes(a.bytes));
  return bits.join("  ·  ");
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes | 0} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
