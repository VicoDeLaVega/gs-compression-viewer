import type { SplatSet } from "./splats";

export type VoxelBounds = {
  min: [number, number, number];
  max: [number, number, number];
};

export type VoxelHorizonField = {
  bounds: VoxelBounds;
  dims: [number, number, number];
  directions: Float32Array;
  horizons: Float32Array;
  opacity: Float32Array;
  maxDistance: number;
  buildMs: number;
};

export type HorizonSample = {
  distance: number;
  voxelIndex: number;
  voxelCoord: [number, number, number];
  directionIndex: number;
  directionDot: number;
};

const DIRECTION_BUCKETS = [
  1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1,
  1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1,
  -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1,
];

function normalizeDirections(src: number[]) {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 3) {
    const x = src[i];
    const y = src[i + 1];
    const z = src[i + 2];
    const invLen = 1 / Math.hypot(x, y, z);
    out[i] = x * invLen;
    out[i + 1] = y * invLen;
    out[i + 2] = z * invLen;
  }
  return out;
}

export const VOXEL_DIRECTIONS = normalizeDirections(DIRECTION_BUCKETS);

function computeWorldBounds(splats: SplatSet, includePoints: Array<{ x: number; y: number; z: number }> = []): VoxelBounds {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < splats.count; i++) {
    const o = i * 3;
    const x = splats.centers[o];
    const y = splats.centers[o + 1];
    const z = splats.centers[o + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  for (const point of includePoints) {
    if (point.x < minX) minX = point.x; if (point.x > maxX) maxX = point.x;
    if (point.y < minY) minY = point.y; if (point.y > maxY) maxY = point.y;
    if (point.z < minZ) minZ = point.z; if (point.z > maxZ) maxZ = point.z;
  }

  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  const sx = maxX - minX || 1;
  const sy = maxY - minY || 1;
  const sz = maxZ - minZ || 1;
  const diagonal = Math.hypot(sx, sy, sz);
  const halfExtent = Math.max(sx, sy, sz, diagonal * 1.6) * 0.5;

  return {
    min: [cx - halfExtent, cy - halfExtent, cz - halfExtent],
    max: [cx + halfExtent, cy + halfExtent, cz + halfExtent],
  };
}

function voxelIndex(x: number, y: number, z: number, nx: number, ny: number) {
  return x + y * nx + z * nx * ny;
}

function pointToVoxel(
  bounds: VoxelBounds,
  dims: [number, number, number],
  x: number,
  y: number,
  z: number,
) {
  const [nx, ny, nz] = dims;
  const fx = (x - bounds.min[0]) / (bounds.max[0] - bounds.min[0]);
  const fy = (y - bounds.min[1]) / (bounds.max[1] - bounds.min[1]);
  const fz = (z - bounds.min[2]) / (bounds.max[2] - bounds.min[2]);
  if (fx < 0 || fx >= 1 || fy < 0 || fy >= 1 || fz < 0 || fz >= 1) return -1;
  const ix = Math.min(nx - 1, Math.floor(fx * nx));
  const iy = Math.min(ny - 1, Math.floor(fy * ny));
  const iz = Math.min(nz - 1, Math.floor(fz * nz));
  return voxelIndex(ix, iy, iz, nx, ny);
}

function voxelCenter(bounds: VoxelBounds, dims: [number, number, number], index: number) {
  const [nx, ny] = dims;
  const iz = Math.floor(index / (nx * ny));
  const rem = index - iz * nx * ny;
  const iy = Math.floor(rem / nx);
  const ix = rem - iy * nx;
  const sx = (bounds.max[0] - bounds.min[0]) / dims[0];
  const sy = (bounds.max[1] - bounds.min[1]) / dims[1];
  const sz = (bounds.max[2] - bounds.min[2]) / dims[2];
  return [
    bounds.min[0] + (ix + 0.5) * sx,
    bounds.min[1] + (iy + 0.5) * sy,
    bounds.min[2] + (iz + 0.5) * sz,
  ];
}

function voxelCoordFromIndex(index: number, dims: [number, number, number]): [number, number, number] {
  const [nx, ny] = dims;
  const iz = Math.floor(index / (nx * ny));
  const rem = index - iz * nx * ny;
  const iy = Math.floor(rem / nx);
  const ix = rem - iy * nx;
  return [ix, iy, iz];
}

export function createVoxelHorizonField(
  splats: SplatSet,
  resolution: number,
  includePoints: Array<{ x: number; y: number; z: number }> = [],
): VoxelHorizonField {
  const t0 = performance.now();
  const dims: [number, number, number] = [resolution, resolution, resolution];
  const bounds = computeWorldBounds(splats, includePoints);
  const [nx, ny, nz] = dims;
  const voxelCount = nx * ny * nz;
  const density = new Float32Array(voxelCount);
  const opacity = new Float32Array(voxelCount);
  const sizeX = (bounds.max[0] - bounds.min[0]) / nx;
  const sizeY = (bounds.max[1] - bounds.min[1]) / ny;
  const sizeZ = (bounds.max[2] - bounds.min[2]) / nz;
  const minCellSize = Math.min(sizeX, sizeY, sizeZ);
  const maxDistance = Math.hypot(
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
  );

  for (let i = 0; i < splats.count; i++) {
    const o3 = i * 3;
    const vi = pointToVoxel(bounds, dims, splats.centers[o3], splats.centers[o3 + 1], splats.centers[o3 + 2]);
    if (vi < 0) continue;
    const alpha = Math.max(0, splats.colors[i * 4 + 3]);
    const radius = (splats.scales[o3] + splats.scales[o3 + 1] + splats.scales[o3 + 2]) / 3;
    density[vi] += alpha * Math.max(0.25, radius / Math.max(1e-6, minCellSize));
  }

  for (let i = 0; i < voxelCount; i++) {
    opacity[i] = 1 - Math.exp(-density[i] * 0.08);
  }

  const dirCount = VOXEL_DIRECTIONS.length / 3;
  const horizons = new Float32Array(voxelCount * dirCount);
  const step = minCellSize * 0.55;
  const threshold = 0.99;

  for (let vi = 0; vi < voxelCount; vi++) {
    const [cx, cy, cz] = voxelCenter(bounds, dims, vi);
    for (let di = 0; di < dirCount; di++) {
      const d3 = di * 3;
      const dx = VOXEL_DIRECTIONS[d3];
      const dy = VOXEL_DIRECTIONS[d3 + 1];
      const dz = VOXEL_DIRECTIONS[d3 + 2];
      let transmittance = 1;
      let horizon = maxDistance;

      for (let dist = 0; dist <= maxDistance; dist += step) {
        const si = pointToVoxel(bounds, dims, cx + dx * dist, cy + dy * dist, cz + dz * dist);
        if (si < 0) continue;
        transmittance *= 1 - opacity[si];
        if (1 - transmittance >= threshold) {
          horizon = dist;
          break;
        }
      }

      horizons[vi * dirCount + di] = horizon;
    }
  }

  return {
    bounds,
    dims,
    directions: VOXEL_DIRECTIONS,
    horizons,
    opacity,
    maxDistance,
    buildMs: performance.now() - t0,
  };
}

export function sampleVoxelHorizon(
  field: VoxelHorizonField,
  cameraPos: { x: number; y: number; z: number },
  viewDir: { x: number; y: number; z: number },
): HorizonSample | null {
  const voxel = pointToVoxel(field.bounds, field.dims, cameraPos.x, cameraPos.y, cameraPos.z);
  if (voxel < 0) return null;

  let bestDirection = 0;
  let bestDot = -Infinity;
  for (let i = 0; i < field.directions.length; i += 3) {
    const dot = (
      viewDir.x * field.directions[i] +
      viewDir.y * field.directions[i + 1] +
      viewDir.z * field.directions[i + 2]
    );
    if (dot > bestDot) {
      bestDot = dot;
      bestDirection = i / 3;
    }
  }

  const dirCount = field.directions.length / 3;
  return {
    distance: field.horizons[voxel * dirCount + bestDirection],
    voxelIndex: voxel,
    voxelCoord: voxelCoordFromIndex(voxel, field.dims),
    directionIndex: bestDirection,
    directionDot: bestDot,
  };
}
