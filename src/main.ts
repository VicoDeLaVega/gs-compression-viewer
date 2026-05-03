import * as THREE from "three";
import "./styles.css";
import { compressSplats, sortSplats, type CompressionConfig, type CompressionMetrics } from "./compression";
import { GaussianRenderer } from "./renderer";
import { SCENES, type SplatSet } from "./splats";
import { LOCAL_SPZ_SCENES, loadLocalSpz } from "./spzLoader";
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
  backend: document.querySelector<HTMLElement>("#m-backend"),
};

const renderer = new GaussianRenderer(canvas);
const clock = new THREE.Clock();

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
let backendMode: "ts" | "wasm" = "ts";
let wasmBackend: WasmBackend | null = null;
let fpsLastTime = performance.now();
let fpsFrames = 0;
let lastFps = 0;

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

async function loadScene(id: string) {
  const localSpz = LOCAL_SPZ_SCENES.find((item) => item.id === id);
  const scene = SCENES.find((item) => item.id === id) ?? SCENES[0];
  reference = localSpz ? await loadLocalSpz(localSpz) : await scene.create();
  renderer.fitToSplats(reference);
  lastSortMs = 0;
  compute();
}

function updateMetrics() {
  const n = reference.count;
  const currentBytes = n * currentMetrics.currentBytesPerSplat;
  const compressedBytes = n * currentMetrics.compressedBytesPerSplat;
  if (metrics.count) metrics.count.textContent = n.toLocaleString();
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
  if (metrics.backend) metrics.backend.textContent = `${currentMetrics.mortonBackend} Morton, ${lastSortBackend} sort`;
}

function updateSorted() {
  if (!active) return;
  const result = backendMode === "wasm" && wasmBackend
    ? wasmBackend.sortSplats(active, renderer.camera.matrixWorldInverse.elements)
    : sortSplats(active, renderer.camera.matrixWorldInverse.elements);
  sorted = result.splats;
  lastSortMs = result.sortMs;
  lastSortBackend = result.backend ?? "TS";
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

backendTsButton?.addEventListener("click", () => {
  setBackendMode("ts");
});

backendWasmButton?.addEventListener("click", () => {
  setBackendMode("wasm");
});

sceneSelect.addEventListener("change", () => {
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
  renderer.controls.update();
  updateSorted();
  const pulse = 1 + Math.sin(clock.getElapsedTime() * 0.7) * 0.02;
  renderer.render(pulse);
  fpsFrames += 1;
  const now = performance.now();
  const elapsed = now - fpsLastTime;
  if (elapsed >= 500) {
    lastFps = (fpsFrames * 1000) / elapsed;
    fpsFrames = 0;
    fpsLastTime = now;
    updateMetrics();
  }
}

frame();
