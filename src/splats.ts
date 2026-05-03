export type SplatSet = {
  name: string;
  count: number;
  centers: Float32Array;
  radii: Float32Array;
  colors: Float32Array;
};

export type SceneFactory = {
  id: string;
  label: string;
  create: () => SplatSet | Promise<SplatSet>;
};

const TAU = Math.PI * 2;

function makeSet(name: string, count: number): SplatSet {
  return {
    name,
    count,
    centers: new Float32Array(count * 3),
    radii: new Float32Array(count),
    colors: new Float32Array(count * 4),
  };
}

function rand(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function writeColor(out: Float32Array, i: number, r: number, g: number, b: number, a: number) {
  const o = i * 4;
  out[o] = r;
  out[o + 1] = g;
  out[o + 2] = b;
  out[o + 3] = a;
}

export function cloneSplats(src: SplatSet, name = src.name): SplatSet {
  return {
    name,
    count: src.count,
    centers: new Float32Array(src.centers),
    radii: new Float32Array(src.radii),
    colors: new Float32Array(src.colors),
  };
}

export const SCENES: SceneFactory[] = [
  {
    id: "helix",
    label: "Synthetic helix - 40k",
    create() {
      const n = 40000;
      const out = makeSet("helix", n);
      const noise = rand(7);
      for (let i = 0; i < n; i++) {
        const t = i / n;
        const arm = i % 3;
        const angle = t * TAU * 8 + arm * (TAU / 3);
        const band = (noise() - 0.5) * 0.28;
        const radius = 1.0 + 0.3 * Math.sin(t * TAU * 5) + band;
        const o3 = i * 3;
        out.centers[o3] = Math.cos(angle) * radius + (noise() - 0.5) * 0.05;
        out.centers[o3 + 1] = (t - 0.5) * 3.2 + (noise() - 0.5) * 0.08;
        out.centers[o3 + 2] = Math.sin(angle) * radius + (noise() - 0.5) * 0.05;
        out.radii[i] = 9 + noise() * 9;
        writeColor(out.colors, i, 0.15 + 0.8 * t, 0.35 + 0.35 * Math.sin(angle), 0.9 - 0.55 * t, 0.42);
      }
      return out;
    },
  },
  {
    id: "volume",
    label: "Dense volume - 60k",
    create() {
      const n = 60000;
      const out = makeSet("volume", n);
      const noise = rand(31);
      for (let i = 0; i < n; i++) {
        const u = noise();
        const v = noise();
        const w = noise();
        const theta = TAU * u;
        const phi = Math.acos(2 * v - 1);
        const r = Math.cbrt(w) * 1.7;
        const x = r * Math.sin(phi) * Math.cos(theta);
        const y = r * Math.cos(phi) * 0.82;
        const z = r * Math.sin(phi) * Math.sin(theta);
        const o3 = i * 3;
        out.centers[o3] = x;
        out.centers[o3 + 1] = y;
        out.centers[o3 + 2] = z;
        out.radii[i] = 7 + noise() * 8;
        writeColor(out.colors, i, 0.25 + 0.35 * (x + 1.7) / 3.4, 0.2 + 0.65 * (y + 1.4) / 2.8, 0.45 + 0.35 * (z + 1.7) / 3.4, 0.28);
      }
      return out;
    },
  },
  {
    id: "layers",
    label: "Transparent layers - 30k",
    create() {
      const n = 30000;
      const out = makeSet("layers", n);
      const noise = rand(97);
      for (let i = 0; i < n; i++) {
        const layer = i % 6;
        const x = (noise() - 0.5) * 4.5;
        const y = (noise() - 0.5) * 2.2;
        const z = (layer - 2.5) * 0.35 + (noise() - 0.5) * 0.08;
        const o3 = i * 3;
        out.centers[o3] = x;
        out.centers[o3 + 1] = y;
        out.centers[o3 + 2] = z;
        out.radii[i] = 10 + noise() * 16;
        writeColor(out.colors, i, 0.18 + layer * 0.11, 0.75 - layer * 0.07, 0.45 + noise() * 0.25, 0.24 + layer * 0.025);
      }
      return out;
    },
  },
];
