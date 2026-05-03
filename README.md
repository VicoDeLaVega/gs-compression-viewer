# GS Compression Viewer

Independent Gaussian Splatting compression playground.

This project intentionally does not use Spark. It is a small WebGL2 + Three.js viewer that loads synthetic splat sets, sorts splats back-to-front, applies clustered quantization, and reports memory/quality metrics.

## Run

From the Spark repo root:

```bash
npx vite gs-compression-viewer --host 0.0.0.0 --port 8082
```

Then open:

```text
http://localhost:8082/
```

## Current Implementation

- WebGL2 instanced Gaussian billboard renderer
- Three.js camera and OrbitControls
- CPU back-to-front sort every camera movement
- Synthetic scenes for quick testing
- Local Spark `.spz` assets copied into `assets/examples/splats`
- Morton-ordered clusters
- Independent SPZ reader for centers, RGB, opacity, and scale-derived billboard radius
- Position quantization at 16/10/8/6 bits
- On/off toggles for position, radius, color, and alpha compression
- Cluster color ramp with 2/4/8 entries
- Optional alpha quantization
- Metrics for bytes/splat, theoretical gain, position p90, color PSNR, and sort time
- TypeScript sort is checked every frame, but recomputed only when the camera matrix changes

## Intended WASM Split

The TypeScript path is a reference implementation. The C++ core in `wasm/` is intended to replace:

- file parsing
- Morton code generation
- cluster quantization/dequantization
- camera depth sort
- metrics

The WebGL upload and renderer should remain unchanged after swapping in the WASM backend.
