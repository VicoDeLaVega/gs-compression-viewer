import { gunzipSync } from "fflate";
import type { SplatSet } from "./splats";

const SH_C0 = 0.28209479177387814;
const LOCAL_SPZ_BASE = "/assets/examples/splats/";

export type SpzScene = {
  id: string;
  label: string;
  file: string;
  maxSplats?: number;
};

export const LOCAL_SPZ_SCENES: SpzScene[] = [
  { id: "spz-robot-head", label: "Spark: robot head - 45k", file: "robot-head.spz" },
  { id: "spz-pad-thai", label: "Spark: pad thai - 95k", file: "food/pad-thai.spz" },
  { id: "spz-penguin", label: "Spark: penguin - 128k", file: "penguin.spz" },
  { id: "spz-fly", label: "Spark: fly - 146k", file: "fly.spz" },
  { id: "spz-butterfly-wings-closed", label: "Spark: butterfly wings closed - 149k", file: "butterfly-wings-closed.spz" },
  { id: "spz-butterfly-ai", label: "Spark: butterfly AI - 168k", file: "butterfly-ai.spz" },
  { id: "spz-butterfly", label: "Spark: butterfly - 177k", file: "butterfly.spz" },
  { id: "spz-dessert", label: "Spark: dessert - 204k", file: "dessert.spz" },
  { id: "spz-cat", label: "Spark: cat - 206k", file: "cat.spz" },
  { id: "spz-gyro", label: "Spark: gyro - 213k", file: "food/gyro.spz" },
  { id: "spz-forge", label: "Spark: forge - 233k", file: "forge.spz" },
  { id: "spz-branzino", label: "Spark: branzino - 280k sampled", file: "food/branzino-amarin.spz", maxSplats: 250000 },
  { id: "spz-fireplace", label: "Spark: fireplace - 301k sampled", file: "fireplace.spz", maxSplats: 250000 },
  { id: "spz-tomahawk", label: "Spark: tomahawk - 311k sampled", file: "food/tomahawk-niku.spz", maxSplats: 250000 },
  { id: "spz-furry-logo", label: "Spark: furry logo pedestal - 322k sampled", file: "furry-logo-pedestal.spz", maxSplats: 250000 },
  { id: "spz-distant-igloo", label: "Spark: distant igloo - 347k sampled", file: "distant-igloo.spz", maxSplats: 250000 },
  { id: "spz-primerib", label: "Spark: primerib - 372k sampled", file: "food/primerib-tamos.spz", maxSplats: 250000 },
  { id: "spz-steaksandwich", label: "Spark: steak sandwich - 439k sampled", file: "food/steaksandwich-mels.spz", maxSplats: 250000 },
  { id: "spz-painted-bedroom", label: "Spark: painted bedroom - 500k sampled", file: "painted-bedroom.spz", maxSplats: 250000 },
  { id: "spz-valley", label: "Spark: valley - 500k sampled", file: "valley.spz", maxSplats: 250000 },
  { id: "spz-woobles", label: "Spark: woobles - 578k sampled", file: "woobles.spz", maxSplats: 250000 },
  { id: "spz-snow-street", label: "Spark: snow street - 982k sampled", file: "snow-street.spz", maxSplats: 250000 },
  { id: "spz-greyscale-bedroom", label: "Spark: greyscale bedroom - 2M sampled", file: "greyscale-bedroom.spz", maxSplats: 250000 },
];

function fromHalf(h: number) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 2 ** 10);
  if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 2 ** 10);
}

function readI24(bytes: Uint8Array, offset: number) {
  return ((bytes[offset + 2] << 24) | (bytes[offset + 1] << 16) | (bytes[offset] << 8)) >> 8;
}

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function robustBounds(centers: Float32Array, count: number, low = 0.02, high = 0.98) {
  const xs = new Array<number>(count);
  const ys = new Array<number>(count);
  const zs = new Array<number>(count);
  for (let i = 0; i < count; i++) {
    const o = i * 3;
    xs[i] = centers[o];
    ys[i] = centers[o + 1];
    zs[i] = centers[o + 2];
  }
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  zs.sort((a, b) => a - b);
  const lo = Math.max(0, Math.min(count - 1, Math.floor(count * low)));
  const hi = Math.max(lo, Math.min(count - 1, Math.floor(count * high)));
  return {
    minX: xs[lo],
    maxX: xs[hi],
    minY: ys[lo],
    maxY: ys[hi],
    minZ: zs[lo],
    maxZ: zs[hi],
  };
}

export async function loadLocalSpz(scene: SpzScene): Promise<SplatSet> {
  const response = await fetch(LOCAL_SPZ_BASE + scene.file);
  if (!response.ok) throw new Error(`Failed to load ${scene.file}: ${response.status}`);
  const compressed = new Uint8Array(await response.arrayBuffer());
  const bytes = gunzipSync(compressed);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(0, true) !== 0x5053474e) throw new Error("Invalid SPZ magic");

  const version = view.getUint32(4, true);
  if (version < 1 || version > 3) throw new Error(`Unsupported SPZ version ${version}`);
  const sourceCount = view.getUint32(8, true);
  const shDegree = bytes[12];
  const fractionalBits = bytes[13];
  const flags = bytes[14];
  const outCount = Math.min(sourceCount, scene.maxSplats ?? sourceCount);
  const stride = Math.max(1, Math.floor(sourceCount / outCount));
  const count = Math.ceil(sourceCount / stride);

  const centers = new Float32Array(count * 3);
  const radii = new Float32Array(count);
  const colors = new Float32Array(count * 4);
  let offset = 16;

  const centerBytes = version === 1 ? sourceCount * 6 : sourceCount * 9;
  for (let i = 0, out = 0; i < sourceCount; i += stride, out++) {
    const dst = out * 3;
    if (version === 1) {
      const src = offset + i * 6;
      centers[dst] = fromHalf(view.getUint16(src, true));
      centers[dst + 1] = fromHalf(view.getUint16(src + 2, true));
      centers[dst + 2] = fromHalf(view.getUint16(src + 4, true));
    } else {
      const fixed = 1 << fractionalBits;
      const src = offset + i * 9;
      centers[dst] = readI24(bytes, src) / fixed;
      centers[dst + 1] = readI24(bytes, src + 3) / fixed;
      centers[dst + 2] = readI24(bytes, src + 6) / fixed;
    }
  }
  offset += centerBytes;

  const bounds = robustBounds(centers, count);
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;
  const centerZ = (bounds.minZ + bounds.maxZ) / 2;
  const diag = Math.hypot(
    bounds.maxX - bounds.minX,
    bounds.maxY - bounds.minY,
    bounds.maxZ - bounds.minZ,
  ) || 1;

  const opacityScale = flags & 0x80 ? 2 : 1;
  for (let i = 0, out = 0; i < sourceCount; i += stride, out++) {
    colors[out * 4 + 3] = clamp((bytes[offset + i] / 255) * opacityScale, 0, 1);
  }
  offset += sourceCount;

  const rgbScale = SH_C0 / 0.15;
  for (let i = 0, out = 0; i < sourceCount; i += stride, out++) {
    const src = offset + i * 3;
    const dst = out * 4;
    colors[dst] = clamp((bytes[src] / 255 - 0.5) * rgbScale + 0.5, 0, 1);
    colors[dst + 1] = clamp((bytes[src + 1] / 255 - 0.5) * rgbScale + 0.5, 0, 1);
    colors[dst + 2] = clamp((bytes[src + 2] / 255 - 0.5) * rgbScale + 0.5, 0, 1);
  }
  offset += sourceCount * 3;

  for (let i = 0, out = 0; i < sourceCount; i += stride, out++) {
    const src = offset + i * 3;
    const sx = Math.exp(bytes[src] / 16 - 10);
    const sy = Math.exp(bytes[src + 1] / 16 - 10);
    const sz = Math.exp(bytes[src + 2] / 16 - 10);
    radii[out] = clamp(((sx + sy + sz) / 3 / diag) * 7000, 2, 34);
  }
  offset += sourceCount * 3;

  // Skip quaternions and SH. This viewer is intentionally billboard-only for now.
  offset += sourceCount * (version === 3 ? 4 : 3);
  const shVecs = [0, 3, 8, 15][shDegree] ?? 0;
  offset += sourceCount * shVecs * 3;

  const scale = 3 / diag;
  for (let i = 0; i < count; i++) {
    const o = i * 3;
    const x = (centers[o] - centerX) * scale;
    const y = (centers[o + 1] - centerY) * scale;
    const z = (centers[o + 2] - centerZ) * scale;
    centers[o] = x;
    centers[o + 1] = -y;
    centers[o + 2] = -z;
  }

  return {
    name: scene.id,
    count,
    centers,
    radii,
    colors,
  };
}
