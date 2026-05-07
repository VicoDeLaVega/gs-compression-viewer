import * as THREE from "three";
import "./styles.css";
import { compressSplats, sortSplats, type CompressionConfig, type CompressionMetrics } from "./compression";
import { GaussianRenderer } from "./renderer";
import { SCENES, type SplatSet } from "./splats";
import { LOCAL_SPZ_SCENES, loadLocalSpz } from "./spzLoader";
import { createVoxelHorizonField, sampleVoxelHorizon, type HorizonSample, type VoxelHorizonField } from "./voxelField";
import { loadWasmBackend, type WasmBackend } from "./wasmBackend";

const canvas = document.querySelector<HTMLCanvasElement>("#viewer");
const sceneSelect = document.querySelector<HTMLSelectElement>("#scene-select");
const clusterSlider = document.querySelector<HTMLInputElement>("#cluster-size");
const clusterValue = document.querySelector<HTMLElement>("#cluster-size-value");
const positionToggle = document.querySelector<HTMLInputElement>("#enable-position");
const scaleToggle = document.querySelector<HTMLInputElement>("#enable-scale");
const colorToggle = document.querySelector<HTMLInputElement>("#enable-color");
const alphaToggle = document.querySelector<HTMLInputElement>("#enable-alpha");
const backendTsButton = document.querySelector<HTMLButtonElement>("#backend-ts");
const backendWasmButton = document.querySelector<HTMLButtonElement>("#backend-wasm");
const stddevSlider  = document.querySelector<HTMLInputElement>("#render-stddev");
const stddevValue   = document.querySelector<HTMLElement>("#render-stddev-value");
const falloffSlider = document.querySelector<HTMLInputElement>("#render-falloff");
const falloffValue  = document.querySelector<HTMLElement>("#render-falloff-value");
const blurSlider    = document.querySelector<HTMLInputElement>("#render-blur");
const blurValue     = document.querySelector<HTMLElement>("#render-blur-value");
const packedAlphaToggle = document.querySelector<HTMLInputElement>("#render-packed-alpha");
const presetSpark   = document.querySelector<HTMLButtonElement>("#preset-spark");
const presetSharp   = document.querySelector<HTMLButtonElement>("#preset-sharp");
const voxelResolutionSlider = document.querySelector<HTMLInputElement>("#voxel-resolution");
const voxelResolutionValue = document.querySelector<HTMLElement>("#voxel-resolution-value");
const horizonMarginSlider = document.querySelector<HTMLInputElement>("#horizon-margin");
const horizonMarginValue = document.querySelector<HTMLElement>("#horizon-margin-value");
const showVoxelGridToggle = document.querySelector<HTMLInputElement>("#show-voxel-grid");
const opacityHorizonToggle = document.querySelector<HTMLInputElement>("#enable-opacity-horizon");
if (!canvas || !sceneSelect || !clusterSlider || !clusterValue || !positionToggle || !scaleToggle || !colorToggle || !alphaToggle) {
  throw new Error("Missing UI elements");
}

const metrics = {
  count: document.querySelector<HTMLElement>("#m-count"),
  current: document.querySelector<HTMLElement>("#m-current"),
  compressed: document.querySelector<HTMLElement>("#m-compressed"),
  bps: document.querySelector<HTMLElement>("#m-bps"),
  gain: document.querySelector<HTMLElement>("#m-gain"),
  posP90: document.querySelector<HTMLElement>("#m-pos-p90"),
  scaleP90: document.querySelector<HTMLElement>("#m-scale-p90"),
  psnr: document.querySelector<HTMLElement>("#m-psnr"),
  fps: document.querySelector<HTMLElement>("#m-fps"),
  morton: document.querySelector<HTMLElement>("#m-morton"),
  sort: document.querySelector<HTMLElement>("#m-sort"),
  voxels: document.querySelector<HTMLElement>("#m-voxels"),
  horizon: document.querySelector<HTMLElement>("#m-horizon"),
  horizonDebug: document.querySelector<HTMLElement>("#m-horizon-debug"),
  backend: document.querySelector<HTMLElement>("#m-backend"),
};

const renderer = new GaussianRenderer(canvas);
const clock = new THREE.Clock();
renderer.controls.enabled = false;

// Render parameter sliders
function applyRenderPreset(stddev: number, falloff: number, blur: number, packedAlphaBoost = false) {
  renderer.maxStdDev  = stddev;
  renderer.falloff    = falloff;
  renderer.blurAmount = blur;
  renderer.packedAlphaBoost = packedAlphaBoost;
  if (stddevSlider)  { stddevSlider.value  = String(stddev);  }
  if (stddevValue)   { stddevValue.textContent  = stddev.toFixed(2); }
  if (falloffSlider) { falloffSlider.value = String(falloff); }
  if (falloffValue)  { falloffValue.textContent  = falloff.toFixed(2); }
  if (blurSlider)    { blurSlider.value    = String(blur);    }
  if (blurValue)     { blurValue.textContent     = blur.toFixed(2); }
  if (packedAlphaToggle) { packedAlphaToggle.checked = packedAlphaBoost; }
}

stddevSlider?.addEventListener("input", () => {
  const v = parseFloat(stddevSlider!.value);
  renderer.maxStdDev = v;
  if (stddevValue) stddevValue.textContent = v.toFixed(2);
});
falloffSlider?.addEventListener("input", () => {
  const v = parseFloat(falloffSlider!.value);
  renderer.falloff = v;
  if (falloffValue) falloffValue.textContent = v.toFixed(2);
});
blurSlider?.addEventListener("input", () => {
  const v = parseFloat(blurSlider!.value);
  renderer.blurAmount = v;
  if (blurValue) blurValue.textContent = v.toFixed(2);
});
packedAlphaToggle?.addEventListener("change", () => {
  renderer.packedAlphaBoost = packedAlphaToggle.checked;
});
presetSpark?.addEventListener("click", () => applyRenderPreset(Math.sqrt(8), 1.0, 0.3));
presetSharp?.addEventListener("click", () => applyRenderPreset(0.43, 0.12, 0.0));

let reference: SplatSet;
let compressed: SplatSet;
let active: SplatSet;
let sorted: SplatSet | null = null;
let mode: "reference" | "compressed" = "compressed";
let config: CompressionConfig = {
  clusterSize: 128,
  enablePosition: true,
  positionBits: 8,
  enableScale: true,
  scaleBits: 8,
  enableColor: true,
  rampColors: 4,
  enableAlpha: true,
};
let lastSortMs = 0;
let lastSortBackend = "TS";
let currentMetrics: CompressionMetrics;
let voxelField: VoxelHorizonField | null = null;
let voxelResolution = 8;
let horizonSafetyMargin = 0.25;
let opacityHorizonEnabled = false;
let lastHorizonSample: HorizonSample | null = null;
let lastRenderedCount = 0;
let lastCulledCount = 0;
let backendMode: "ts" | "wasm" = "ts";
let wasmBackend: WasmBackend | null = null;
let fpsLastTime = performance.now();
let fpsFrames = 0;
let lastFps = 0;
let lastFrameTime = performance.now();
let navigationSpeed = 1.0;
let lookYaw = 0;
let lookPitch = 0;
let isMouseLooking = false;
let lastPointerX = 0;
let lastPointerY = 0;

const movementKeys = new Set<string>();
const movementForward = new THREE.Vector3();
const movementRight = new THREE.Vector3();
const movementUp = new THREE.Vector3(0, 1, 0);
const movementDelta = new THREE.Vector3();
const lookEuler = new THREE.Euler(0, 0, 0, "YXZ");
const viewDirection = new THREE.Vector3();

function isTextEntryTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(target.tagName);
}

window.addEventListener("keydown", (event) => {
  if (isTextEntryTarget(event.target)) return;
  const key = event.key.toLowerCase();
  if (!["w", "a", "s", "d", "q", "e"].includes(key)) return;
  movementKeys.add(key);
  event.preventDefault();
});

window.addEventListener("keyup", (event) => {
  movementKeys.delete(event.key.toLowerCase());
});

window.addEventListener("blur", () => {
  movementKeys.clear();
  isMouseLooking = false;
});

function syncLookAnglesFromCamera() {
  lookEuler.setFromQuaternion(renderer.camera.quaternion, "YXZ");
  lookPitch = lookEuler.x;
  lookYaw = lookEuler.y;
}

function applyLookAngles() {
  const maxPitch = Math.PI / 2 - 0.01;
  lookPitch = Math.max(-maxPitch, Math.min(maxPitch, lookPitch));
  lookEuler.set(lookPitch, lookYaw, 0, "YXZ");
  renderer.camera.quaternion.setFromEuler(lookEuler);
  renderer.camera.updateMatrixWorld();
}

canvas.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  isMouseLooking = true;
  lastPointerX = event.clientX;
  lastPointerY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
  event.preventDefault();
});

canvas.addEventListener("pointermove", (event) => {
  if (!isMouseLooking) return;
  const dx = event.movementX || event.clientX - lastPointerX;
  const dy = event.movementY || event.clientY - lastPointerY;
  lastPointerX = event.clientX;
  lastPointerY = event.clientY;

  const sensitivity = 0.0025;
  lookYaw -= dx * sensitivity;
  lookPitch -= dy * sensitivity;
  applyLookAngles();
});

canvas.addEventListener("pointerup", (event) => {
  isMouseLooking = false;
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
});

canvas.addEventListener("pointercancel", () => {
  isMouseLooking = false;
});

function updateKeyboardNavigation(deltaSeconds: number) {
  if (movementKeys.size === 0) return;

  renderer.camera.getWorldDirection(movementForward);
  movementRight.copy(movementForward).cross(renderer.camera.up).normalize();
  movementDelta.set(0, 0, 0);

  if (movementKeys.has("w")) movementDelta.add(movementForward);
  if (movementKeys.has("s")) movementDelta.sub(movementForward);
  if (movementKeys.has("d")) movementDelta.add(movementRight);
  if (movementKeys.has("a")) movementDelta.sub(movementRight);
  if (movementKeys.has("e")) movementDelta.add(movementUp);
  if (movementKeys.has("q")) movementDelta.sub(movementUp);
  if (movementDelta.lengthSq() === 0) return;

  movementDelta.normalize();
  const speed = navigationSpeed;
  const step = speed * deltaSeconds;

  renderer.camera.position.addScaledVector(movementDelta, step);
  renderer.camera.updateMatrixWorld();
}

for (const scene of SCENES) {
  const option = document.createElement("option");
  option.value = scene.id;
  option.textContent = scene.label;
  sceneSelect.appendChild(option);
}
for (const scene of LOCAL_SPZ_SCENES) {
  const option = document.createElement("option");
  option.value = scene.id;
  option.textContent = scene.label;
  sceneSelect.appendChild(option);
}

function fmtBytes(bytes: number) {
  if (bytes < 1024) return `${bytes.toFixed(0)} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MiB`;
}

function fmtPct(v: number) {
  return `${(v * 100).toFixed(1)} %`;
}

function fmtSmallPct(v: number) {
  const pct = v * 100;
  return `${pct < 1 ? pct.toFixed(3) : pct.toFixed(1)} %`;
}

function setButtonGroup(selector: string, attr: string, value: string) {
  document.querySelectorAll<HTMLButtonElement>(selector).forEach((button) => {
    button.classList.toggle("active", button.dataset[attr] === value);
  });
}

function compute() {
  const result = compressSplats(reference, config, backendMode === "wasm" ? wasmBackend ?? undefined : undefined);
  compressed = result.splats;
  currentMetrics = result.metrics;
  active = mode === "reference" ? reference : compressed;
  sorted = null;
  updateMetrics();
}

function rebuildVoxelField() {
  if (!reference) return;
  voxelField = createVoxelHorizonField(reference, voxelResolution, [renderer.camera.position]);
  renderer.uploadVoxelGrid(voxelField.bounds, voxelField.dims);
  lastHorizonSample = null;
  lastRenderedCount = 0;
  lastCulledCount = 0;
  sorted = null;
  if (currentMetrics) updateMetrics();
}

async function loadScene(id: string) {
  movementKeys.clear();
  const localSpz = LOCAL_SPZ_SCENES.find((item) => item.id === id);
  const scene = SCENES.find((item) => item.id === id) ?? SCENES[0];
  reference = localSpz ? await loadLocalSpz(localSpz) : await scene.create();
  renderer.shScale = reference.shScale ?? 1.0;
  renderer.fitToSplats(reference);
  renderer.camera.lookAt(renderer.controls.target);
  renderer.camera.updateMatrixWorld();
  syncLookAnglesFromCamera();
  navigationSpeed = Math.max(0.25, renderer.camera.position.distanceTo(renderer.controls.target) * 0.9);
  lastSortMs = 0;
  rebuildVoxelField();
  compute();
  canvas.focus();
}

function updateMetrics() {
  const n = reference.count;
  const activeCount = active?.count ?? n;
  const currentBytes = n * currentMetrics.currentBytesPerSplat;
  const compressedBytes = n * currentMetrics.compressedBytesPerSplat;
  if (metrics.count) {
    const hasCullStats = opacityHorizonEnabled && (lastRenderedCount > 0 || lastCulledCount > 0);
    metrics.count.textContent = hasCullStats
      ? `${lastRenderedCount.toLocaleString()} / ${activeCount.toLocaleString()}`
      : activeCount.toLocaleString();
  }
  if (metrics.current) metrics.current.textContent = fmtBytes(currentBytes);
  if (metrics.compressed) metrics.compressed.textContent = fmtBytes(compressedBytes);
  if (metrics.bps) metrics.bps.textContent = `${currentMetrics.compressedBytesPerSplat.toFixed(2)} B`;
  if (metrics.gain) metrics.gain.textContent = fmtPct(currentMetrics.gain);
  if (metrics.posP90) metrics.posP90.textContent = currentMetrics.posP90 == null ? "off" : currentMetrics.posP90.toFixed(5);
  if (metrics.scaleP90) metrics.scaleP90.textContent = currentMetrics.scaleP90 == null ? "off" : fmtSmallPct(currentMetrics.scaleP90);
  if (metrics.psnr) metrics.psnr.textContent = currentMetrics.colorPsnr == null ? "off" : `${currentMetrics.colorPsnr.toFixed(2)} dB`;
  if (metrics.fps) metrics.fps.textContent = lastFps > 0 ? lastFps.toFixed(1) : "-";
  if (metrics.morton) metrics.morton.textContent = `${currentMetrics.mortonMs.toFixed(2)} ms`;
  if (metrics.sort) metrics.sort.textContent = `${lastSortMs.toFixed(2)} ms`;
  if (metrics.voxels) {
    metrics.voxels.textContent = voxelField
      ? `${voxelField.dims[0]}³ × ${voxelField.directions.length / 3} dirs (${voxelField.buildMs.toFixed(1)} ms)`
      : "-";
  }
  if (metrics.horizon) {
    if (!opacityHorizonEnabled) {
      metrics.horizon.textContent = "off";
    } else if (!lastHorizonSample) {
      metrics.horizon.textContent = "outside grid";
    } else {
      metrics.horizon.textContent = `${lastCulledCount.toLocaleString()} culled, ${lastHorizonSample.distance.toFixed(2)} d`;
    }
  }
  if (metrics.horizonDebug) {
    if (!lastHorizonSample) {
      metrics.horizonDebug.textContent = "-";
    } else {
      const [vx, vy, vz] = lastHorizonSample.voxelCoord;
      metrics.horizonDebug.textContent = `v ${vx},${vy},${vz} dir ${lastHorizonSample.directionIndex} dot ${lastHorizonSample.directionDot.toFixed(2)}`;
    }
  }
  if (metrics.backend) metrics.backend.textContent = `${currentMetrics.mortonBackend} Morton, ${lastSortBackend} sort`;
}

function updateSorted() {
  if (!active) return;
  renderer.camera.getWorldDirection(viewDirection);
  lastHorizonSample = opacityHorizonEnabled && voxelField
    ? sampleVoxelHorizon(voxelField, renderer.camera.position, viewDirection)
    : null;
  const maxViewDepth = lastHorizonSample
    ? lastHorizonSample.distance + voxelField!.maxDistance * horizonSafetyMargin
    : undefined;
  const result = maxViewDepth == null && backendMode === "wasm" && wasmBackend
    ? wasmBackend.sortSplats(active, renderer.camera.matrixWorldInverse.elements)
    : sortSplats(active, renderer.camera.matrixWorldInverse.elements, { maxViewDepth });
  sorted = result.splats;
  lastSortMs = result.sortMs;
  lastSortBackend = result.backend ?? "TS";
  lastRenderedCount = result.renderedCount ?? sorted.count;
  lastCulledCount = result.culledCount ?? Math.max(0, active.count - lastRenderedCount);
  renderer.upload(sorted);
  updateMetrics();
}

function setBackendMode(nextMode: "ts" | "wasm") {
  if (nextMode === "wasm" && !wasmBackend) return;
  backendMode = nextMode;
  backendTsButton?.classList.toggle("active", backendMode === "ts");
  backendWasmButton?.classList.toggle("active", backendMode === "wasm");
  lastSortMs = 0;
  lastSortBackend = backendMode === "wasm" ? "WASM C++" : "TS";
  compute();
}

document.querySelector("#mode-reference")?.addEventListener("click", () => {
  mode = "reference";
  active = reference;
  sorted = null;
  setButtonGroup("#mode-reference, #mode-compressed", "", "");
  document.querySelector("#mode-reference")?.classList.add("active");
  document.querySelector("#mode-compressed")?.classList.remove("active");
});

document.querySelector("#mode-compressed")?.addEventListener("click", () => {
  mode = "compressed";
  active = compressed;
  sorted = null;
  document.querySelector("#mode-compressed")?.classList.add("active");
  document.querySelector("#mode-reference")?.classList.remove("active");
});

document.querySelectorAll<HTMLButtonElement>("#position-bits .button").forEach((button) => {
  button.addEventListener("click", () => {
    config.positionBits = Number(button.dataset.bits);
    setButtonGroup("#position-bits .button", "bits", String(config.positionBits));
    compute();
  });
});

document.querySelectorAll<HTMLButtonElement>("#scale-bits .button").forEach((button) => {
  button.addEventListener("click", () => {
    config.scaleBits = Number(button.dataset.bits);
    setButtonGroup("#scale-bits .button", "bits", String(config.scaleBits));
    compute();
  });
});

document.querySelectorAll<HTMLButtonElement>("#color-ramp .button").forEach((button) => {
  button.addEventListener("click", () => {
    config.rampColors = Number(button.dataset.colors);
    setButtonGroup("#color-ramp .button", "colors", String(config.rampColors));
    compute();
  });
});

clusterSlider.addEventListener("input", () => {
  config.clusterSize = Number(clusterSlider.value);
  clusterValue.textContent = String(config.clusterSize);
  compute();
});

positionToggle.addEventListener("change", () => {
  config.enablePosition = positionToggle.checked;
  compute();
});

scaleToggle.addEventListener("change", () => {
  config.enableScale = scaleToggle.checked;
  compute();
});

colorToggle.addEventListener("change", () => {
  config.enableColor = colorToggle.checked;
  compute();
});

alphaToggle.addEventListener("change", () => {
  config.enableAlpha = alphaToggle.checked;
  compute();
});

voxelResolutionSlider?.addEventListener("input", () => {
  voxelResolution = Number(voxelResolutionSlider.value);
  if (voxelResolutionValue) voxelResolutionValue.textContent = String(voxelResolution);
});

voxelResolutionSlider?.addEventListener("change", () => {
  rebuildVoxelField();
});

horizonMarginSlider?.addEventListener("input", () => {
  horizonSafetyMargin = Number(horizonMarginSlider.value) / 100;
  if (horizonMarginValue) horizonMarginValue.textContent = `${horizonMarginSlider.value}%`;
});

showVoxelGridToggle?.addEventListener("change", () => {
  renderer.showVoxelGrid = showVoxelGridToggle.checked;
});

opacityHorizonToggle?.addEventListener("change", () => {
  opacityHorizonEnabled = opacityHorizonToggle.checked;
  sorted = null;
  lastHorizonSample = null;
  lastCulledCount = 0;
});

backendTsButton?.addEventListener("click", () => {
  setBackendMode("ts");
});

backendWasmButton?.addEventListener("click", () => {
  setBackendMode("wasm");
});

sceneSelect.addEventListener("change", () => {
  sceneSelect.blur();
  void loadScene(sceneSelect.value);
});

void loadScene(SCENES[0].id);
void loadWasmBackend()
  .then((backend) => {
    wasmBackend = backend;
    if (backendWasmButton) backendWasmButton.disabled = false;
    if (metrics.backend) metrics.backend.textContent = "TS Morton, TS sort";
  })
  .catch((error) => {
    console.warn("WASM backend unavailable", error);
    if (backendWasmButton) backendWasmButton.textContent = "WASM unavailable";
  });

function frame() {
  requestAnimationFrame(frame);
  const now = performance.now();
  const deltaSeconds = Math.min(0.05, (now - lastFrameTime) / 1000);
  lastFrameTime = now;
  updateKeyboardNavigation(deltaSeconds);
  if (renderer.controls.enabled) renderer.controls.update();
  updateSorted();
  const pulse = 1 + Math.sin(clock.getElapsedTime() * 0.7) * 0.02;
  renderer.render(pulse);
  fpsFrames += 1;
  const elapsed = now - fpsLastTime;
  if (elapsed >= 500) {
    lastFps = (fpsFrames * 1000) / elapsed;
    fpsFrames = 0;
    fpsLastTime = now;
    updateMetrics();
  }
}

frame();
