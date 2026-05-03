import type { SplatSet } from "./splats";
import { cloneSplats } from "./splats";

export type CompressionConfig = {
  clusterSize: number;
  enablePosition: boolean;
  positionBits: number;
  enableScale: boolean;
  scaleBits: number;
  enableColor: boolean;
  rampColors: number;
  enableAlpha: boolean;
};

export type CompressionMetrics = {
  currentBytesPerSplat: number;
  compressedBytesPerSplat: number;
  gain: number;
  posP90: number | null;
  scaleP90: number | null;
  colorPsnr: number | null;
  mortonMs: number;
  mortonBackend: string;
};

export type SortResult = {
  splats: SplatSet;
  sortMs: number;
  backend?: string;
};

export type MortonBackend = {
  label: string;
  mortonCodes: (
    centers: Float32Array,
    count: number,
    minX: number,
    minY: number,
    minZ: number,
    invX: number,
    invY: number,
    invZ: number,
  ) => Uint32Array;
};

function expandBits(v: number) {
  v &= 0x3ff;
  v = (v | (v << 16)) & 0x030000ff;
  v = (v | (v << 8)) & 0x0300f00f;
  v = (v | (v << 4)) & 0x030c30c3;
  v = (v | (v << 2)) & 0x09249249;
  return v >>> 0;
}

function morton3(x: number, y: number, z: number) {
  return (expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2)) >>> 0;
}

function percentile(values: Float32Array, p: number) {
  const copy = Array.from(values);
  copy.sort((a, b) => a - b);
  return copy[Math.min(copy.length - 1, Math.floor(copy.length * p))] ?? 0;
}

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v));
}

function estimateBytesPerSplat(cfg: CompressionConfig) {
  const metadataBytes = 4;
  const pos = cfg.enablePosition && cfg.positionBits < 16
    ? (cfg.positionBits * 3) / 8 + (6 * metadataBytes) / cfg.clusterSize
    : 6;
  const scale = cfg.enableScale && cfg.scaleBits <= 8
    ? (cfg.scaleBits * 3) / 8 + (6 * metadataBytes) / cfg.clusterSize
    : 3;
  const quat = 4;
  const rgbIndex = cfg.enableColor ? 1 : 3;
  const rgbRamp = cfg.enableColor ? (cfg.rampColors * 3) / cfg.clusterSize : 0;
  const alpha = cfg.enableAlpha ? 0.5 + (2 * metadataBytes) / cfg.clusterSize : 1;
  return pos + scale + quat + rgbIndex + rgbRamp + alpha;
}

export function compressSplats(
  src: SplatSet,
  cfg: CompressionConfig,
  backend?: MortonBackend,
): { splats: SplatSet; metrics: CompressionMetrics } {
  const out = cloneSplats(src, `${src.name}-compressed`);
  const n = src.count;
  const keys = new Uint32Array(n);
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let i = 0; i < n; i++) {
    const o = i * 3;
    const x = src.centers[o], y = src.centers[o + 1], z = src.centers[o + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }

  const rx = maxX - minX || 1;
  const ry = maxY - minY || 1;
  const rz = maxZ - minZ || 1;
  const mortonT0 = performance.now();
  if (backend) {
    keys.set(backend.mortonCodes(src.centers, n, minX, minY, minZ, 1 / rx, 1 / ry, 1 / rz));
  } else {
    for (let i = 0; i < n; i++) {
      const o = i * 3;
      keys[i] = morton3(
        Math.round(((src.centers[o] - minX) / rx) * 1023),
        Math.round(((src.centers[o + 1] - minY) / ry) * 1023),
        Math.round(((src.centers[o + 2] - minZ) / rz) * 1023),
      );
    }
  }
  const mortonMs = performance.now() - mortonT0;

  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => keys[a] - keys[b]);
  const posErrors = new Float32Array(n);
  const scaleErrors = new Float32Array(n);
  let colorMse = 0;
  const posLevels = 2 ** cfg.positionBits - 1;
  const scaleLevels = 2 ** cfg.scaleBits - 1;

  for (let c0 = 0; c0 < n; c0 += cfg.clusterSize) {
    const c1 = Math.min(n, c0 + cfg.clusterSize);
    let cMinX = Infinity, cMinY = Infinity, cMinZ = Infinity;
    let cMaxX = -Infinity, cMaxY = -Infinity, cMaxZ = -Infinity;
    let sMinX = Infinity, sMinY = Infinity, sMinZ = Infinity;
    let sMaxX = -Infinity, sMaxY = -Infinity, sMaxZ = -Infinity;
    let aMin = 1, aMax = 0;
    const avg = [0, 0, 0];

    for (let j = c0; j < c1; j++) {
      const i = order[j];
      const o3 = i * 3, o4 = i * 4;
      const x = src.centers[o3], y = src.centers[o3 + 1], z = src.centers[o3 + 2];
      if (x < cMinX) cMinX = x; if (x > cMaxX) cMaxX = x;
      if (y < cMinY) cMinY = y; if (y > cMaxY) cMaxY = y;
      if (z < cMinZ) cMinZ = z; if (z > cMaxZ) cMaxZ = z;
      const sx = src.scales[o3], sy = src.scales[o3 + 1], sz = src.scales[o3 + 2];
      if (sx < sMinX) sMinX = sx; if (sx > sMaxX) sMaxX = sx;
      if (sy < sMinY) sMinY = sy; if (sy > sMaxY) sMaxY = sy;
      if (sz < sMinZ) sMinZ = sz; if (sz > sMaxZ) sMaxZ = sz;
      avg[0] += src.colors[o4]; avg[1] += src.colors[o4 + 1]; avg[2] += src.colors[o4 + 2];
      const a = src.colors[o4 + 3];
      if (a < aMin) aMin = a;
      if (a > aMax) aMax = a;
    }

    const inv = 1 / (c1 - c0);
    avg[0] *= inv; avg[1] *= inv; avg[2] *= inv;
    const dir = [0.577, 0.577, 0.577];
    const ramp: number[][] = [];
    for (let k = 0; k < cfg.rampColors; k++) {
      const t = cfg.rampColors === 1 ? 0.5 : k / (cfg.rampColors - 1);
      ramp.push([
        clamp01(avg[0] + (t - 0.5) * dir[0]),
        clamp01(avg[1] + (t - 0.5) * dir[1]),
        clamp01(avg[2] + (t - 0.5) * dir[2]),
      ]);
    }

    const crx = cMaxX - cMinX || 1e-9;
    const cry = cMaxY - cMinY || 1e-9;
    const crz = cMaxZ - cMinZ || 1e-9;
    const srx = sMaxX - sMinX || 1e-9;
    const sry = sMaxY - sMinY || 1e-9;
    const srz = sMaxZ - sMinZ || 1e-9;
    const ar = aMax - aMin || 1e-9;
    for (let j = c0; j < c1; j++) {
      const i = order[j];
      const o3 = i * 3, o4 = i * 4;
      const ox = src.centers[o3], oy = src.centers[o3 + 1], oz = src.centers[o3 + 2];
      if (cfg.enablePosition && cfg.positionBits < 16) {
        const qx = Math.round(((ox - cMinX) / crx) * posLevels);
        const qy = Math.round(((oy - cMinY) / cry) * posLevels);
        const qz = Math.round(((oz - cMinZ) / crz) * posLevels);
        out.centers[o3] = cMinX + (qx / posLevels) * crx;
        out.centers[o3 + 1] = cMinY + (qy / posLevels) * cry;
        out.centers[o3 + 2] = cMinZ + (qz / posLevels) * crz;
      }
      const dx = out.centers[o3] - ox, dy = out.centers[o3 + 1] - oy, dz = out.centers[o3 + 2] - oz;
      posErrors[i] = Math.hypot(dx, dy, dz);

      if (cfg.enableScale && cfg.scaleBits <= 8) {
        const qsx = Math.round(((src.scales[o3] - sMinX) / srx) * scaleLevels);
        const qsy = Math.round(((src.scales[o3 + 1] - sMinY) / sry) * scaleLevels);
        const qsz = Math.round(((src.scales[o3 + 2] - sMinZ) / srz) * scaleLevels);
        out.scales[o3] = sMinX + (qsx / scaleLevels) * srx;
        out.scales[o3 + 1] = sMinY + (qsy / scaleLevels) * sry;
        out.scales[o3 + 2] = sMinZ + (qsz / scaleLevels) * srz;
      }
      const dsx = out.scales[o3] - src.scales[o3];
      const dsy = out.scales[o3 + 1] - src.scales[o3 + 1];
      const dsz = out.scales[o3 + 2] - src.scales[o3 + 2];
      const srcScaleLen = Math.hypot(src.scales[o3], src.scales[o3 + 1], src.scales[o3 + 2]) || 1;
      scaleErrors[i] = Math.hypot(dsx, dsy, dsz) / srcScaleLen;

      let bestD = 0;
      if (cfg.enableColor) {
        let best = ramp[0];
        bestD = Infinity;
        for (const color of ramp) {
          const d = (src.colors[o4] - color[0]) ** 2 + (src.colors[o4 + 1] - color[1]) ** 2 + (src.colors[o4 + 2] - color[2]) ** 2;
          if (d < bestD) { bestD = d; best = color; }
        }
        out.colors[o4] = best[0];
        out.colors[o4 + 1] = best[1];
        out.colors[o4 + 2] = best[2];
      }
      if (cfg.enableAlpha) {
        const qa = Math.round(((src.colors[o4 + 3] - aMin) / ar) * 15);
        out.colors[o4 + 3] = aMin + (qa / 15) * ar;
      }
      colorMse += bestD / 3;
    }
  }

  colorMse /= n;
  const compressedBytesPerSplat = estimateBytesPerSplat(cfg);
  const currentBytesPerSplat = 17;
  return {
    splats: out,
    metrics: {
      currentBytesPerSplat,
      compressedBytesPerSplat,
      gain: 1 - compressedBytesPerSplat / currentBytesPerSplat,
      posP90: cfg.enablePosition && cfg.positionBits < 16 ? percentile(posErrors, 0.9) : null,
      scaleP90: cfg.enableScale && cfg.scaleBits <= 8 ? percentile(scaleErrors, 0.9) : null,
      colorPsnr: cfg.enableColor && colorMse > 0 ? 10 * Math.log10(1 / colorMse) : null,
      mortonMs,
      mortonBackend: backend?.label ?? "TS",
    },
  };
}

export function sortSplats(src: SplatSet, viewMatrix: Float32Array | number[]): SortResult {
  const t0 = performance.now();
  const n = src.count;
  const indices = Array.from({ length: n }, (_, i) => i);
  const e = viewMatrix;
  indices.sort((a, b) => {
    const ao = a * 3, bo = b * 3;
    const az = e[2] * src.centers[ao] + e[6] * src.centers[ao + 1] + e[10] * src.centers[ao + 2] + e[14];
    const bz = e[2] * src.centers[bo] + e[6] * src.centers[bo + 1] + e[10] * src.centers[bo + 2] + e[14];
    return az - bz;
  });

  const sorted = {
    name: `${src.name}-sorted`,
    count: n,
    centers: new Float32Array(n * 3),
    scales: new Float32Array(n * 3),
    quats: new Float32Array(n * 4),
    colors: new Float32Array(n * 4),
  };
  for (let dst = 0; dst < n; dst++) {
    const srcIdx = indices[dst];
    sorted.centers.set(src.centers.subarray(srcIdx * 3, srcIdx * 3 + 3), dst * 3);
    sorted.scales.set(src.scales.subarray(srcIdx * 3, srcIdx * 3 + 3), dst * 3);
    sorted.quats.set(src.quats.subarray(srcIdx * 4, srcIdx * 4 + 4), dst * 4);
    sorted.colors.set(src.colors.subarray(srcIdx * 4, srcIdx * 4 + 4), dst * 4);
  }

  return { splats: sorted, sortMs: performance.now() - t0 };
}
