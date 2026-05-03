import type { MortonBackend, SortResult } from "./compression";
import type { SplatSet } from "./splats";
import wasmUrl from "./wasm/gs_core.wasm?url";

type GsCoreExports = WebAssembly.Exports & {
  d: WebAssembly.Memory;
  f: (
    centersPtr: number,
    count: number,
    minX: number,
    minY: number,
    minZ: number,
    invX: number,
    invY: number,
    invZ: number,
    outCodesPtr: number,
  ) => void;
  g: (centersPtr: number, count: number, viewMatrixPtr: number, outIndicesPtr: number) => void;
  h: (bytes: number) => number;
  i: (ptr: number) => void;
};

export type WasmBackend = MortonBackend & {
  sortSplats: (src: SplatSet, viewMatrix: Float32Array | number[]) => SortResult;
};

function align4(bytes: number) {
  return (bytes + 3) & ~3;
}

function sortedFromIndices(src: SplatSet, indices: Uint32Array): SplatSet {
  const n = src.count;
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

  return sorted;
}

export async function loadWasmBackend(): Promise<WasmBackend> {
  let memory: WebAssembly.Memory | null = null;
  const imports = {
    a: {
      a() {
        throw new Error("WASM aborted");
      },
      b(requestedSize: number) {
        if (!memory) return 0;
        const currentBytes = memory.buffer.byteLength;
        if (requestedSize <= currentBytes) return 1;
        const pages = Math.ceil((requestedSize - currentBytes) / 65536);
        try {
          memory.grow(pages);
          return 1;
        } catch {
          return 0;
        }
      },
      c() {
        throw new Error("WASM C++ exception");
      },
    },
  };

  const response = await fetch(wasmUrl);
  const result = await WebAssembly.instantiate(await response.arrayBuffer(), imports);
  const exports = result.instance.exports as GsCoreExports;
  memory = exports.d;

  function heapF32() {
    return new Float32Array(memory!.buffer);
  }

  function heapU32() {
    return new Uint32Array(memory!.buffer);
  }

  function malloc(bytes: number) {
    const ptr = exports.h(align4(bytes));
    if (!ptr) throw new Error(`WASM malloc failed for ${bytes} bytes`);
    return ptr;
  }

  return {
    label: "WASM C++",

    mortonCodes(centers, count, minX, minY, minZ, invX, invY, invZ) {
      const centersPtr = malloc(centers.byteLength);
      const outPtr = malloc(count * 4);
      try {
        heapF32().set(centers, centersPtr >> 2);
        exports.f(centersPtr, count, minX, minY, minZ, invX, invY, invZ, outPtr);
        return new Uint32Array(heapU32().subarray(outPtr >> 2, (outPtr >> 2) + count));
      } finally {
        exports.i(outPtr);
        exports.i(centersPtr);
      }
    },

    sortSplats(src, viewMatrix) {
      const t0 = performance.now();
      const centersPtr = malloc(src.centers.byteLength);
      const viewPtr = malloc(16 * 4);
      const outPtr = malloc(src.count * 4);
      try {
        heapF32().set(src.centers, centersPtr >> 2);
        heapF32().set(viewMatrix, viewPtr >> 2);
        exports.g(centersPtr, src.count, viewPtr, outPtr);
        const indices = new Uint32Array(heapU32().subarray(outPtr >> 2, (outPtr >> 2) + src.count));
        return {
          splats: sortedFromIndices(src, indices),
          sortMs: performance.now() - t0,
          backend: "WASM C++",
        };
      } finally {
        exports.i(outPtr);
        exports.i(viewPtr);
        exports.i(centersPtr);
      }
    },
  };
}
