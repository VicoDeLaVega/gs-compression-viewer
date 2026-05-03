# WASM Core

`gs_core.cpp` is the planned C++ backend for the viewer.

The current machine does not have Emscripten installed, so this folder is not compiled automatically yet. Once `emcc` is available, the intended output is an ES module exposing functions for sorting, Morton clustering, compression, decompression, and metrics.

Suggested build shape:

```bash
emcc gs_core.cpp \
  -O3 \
  -std=c++20 \
  -s MODULARIZE=1 \
  -s EXPORT_ES6=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s EXPORTED_FUNCTIONS='["_malloc","_free","_sort_depth","_morton_codes"]' \
  -o ../src/wasm/gs_core.js
```

