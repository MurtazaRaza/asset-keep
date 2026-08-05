// Hearing a sound without opening anything.
//
// Quick Look has played audio since M2, and it is the wrong instrument for the
// job it was doing. Auditioning is not previewing one file, it is going through
// a folder of a hundred and thirty sound effects looking for the right footstep,
// and a fullscreen modal that has to be opened and closed around each one turns
// that into three keystrokes per sound and a screen that flashes black between
// every pair of them.
//
// So audition is a mode rather than an action. `p` starts it on the asset under
// the cursor, and while it is on, moving the cursor plays whatever it lands on.
// Arrowing along a row of sound effects then plays them in turn, which is the
// actual activity. `p` again, Escape, or moving onto something that is not audio
// ends it - the last of those because a mode that stays armed while you browse
// a wall of sprites is a mode that will surprise someone later.

import { state, emit } from "./state.js";
import * as api from "./api.js";

// One element for the whole page. A second sound starting cuts the first off,
// which is what auditioning means: comparing two footsteps by hearing them one
// after the other, not both at once.
let player = null;
let active = false;

export const isActive = () => active;
export const playingId = () => (active && player && !player.paused ? current : null);

let current = null;

export function init() {
  player = new Audio();
  player.preload = "none";
  // Not `loop`. A looping footstep is a nuisance, and for the ambience tracks
  // that are meant to loop, the file ending is itself the information: a loop
  // point that does not match is audible exactly at the seam.
  player.addEventListener("ended", () => {
    current = null;
    emit("playing");
  });
  player.addEventListener("error", stop);
}

// Start, stop, or move the mode onto a different asset.
export function toggle(asset) {
  if (active && current === (asset && asset.id)) {
    stop();
    return;
  }
  if (!isAudio(asset)) return;
  active = true;
  play(asset);
}

// Called on every cursor move. Silent unless the mode is already on, which is
// what keeps `p` from turning the grid into a soundboard for the rest of the
// session.
export function follow(asset) {
  if (!active) return;
  if (!isAudio(asset)) {
    stop();
    return;
  }
  play(asset);
}

export function stop() {
  if (!player) return;
  player.pause();
  // Releasing the source as well, so that a paused audition is not still
  // holding a file handle open on a network volume.
  player.removeAttribute("src");
  player.load();
  active = false;
  current = null;
  emit("playing");
}

function play(asset) {
  current = asset.id;
  player.src = api.fileUrl(asset);
  // A rejected promise here is the browser's autoplay policy, which only ever
  // fires before the first real interaction with the page. Nothing to do about
  // it beyond not throwing: the next keypress is itself the interaction that
  // makes the one after it work.
  const started = player.play();
  if (started && started.catch) started.catch(() => {});
  emit("playing");
}

export function isAudio(asset) {
  return Boolean(asset) && asset.kind === "audio";
}

// Whether the current results contain anything audible at all, which is what
// decides if the status bar mentions the key.
export function anyAudio() {
  return state.assets.some(isAudio);
}
