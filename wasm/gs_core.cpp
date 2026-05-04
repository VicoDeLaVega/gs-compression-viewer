/**
 * gs_core.cpp — WASM backend for the Gaussian splat compression viewer.
 *
 * Exported functions
 * ------------------
 *   morton_codes  – compute 30-bit Morton codes for an array of 3-D points
 *   sort_depth    – sort splat indices back-to-front by view-space Z using
 *                   a fast 4-pass 8-bit radix sort (O(n), ~3-5x faster than
 *                   the std::sort-based implementation for large splat counts)
 *
 * Build (requires Emscripten >= 3.1)
 * ------------------------------------
 *   emcc gs_core.cpp \
 *     -O3 -std=c++20 \
 *     -s MODULARIZE=1 \
 *     -s EXPORT_ES6=1 \
 *     -s ALLOW_MEMORY_GROWTH=1 \
 *     -s EXPORTED_FUNCTIONS='["_malloc","_free","_sort_depth","_morton_codes"]' \
 *     -s EXPORTED_RUNTIME_METHODS='[]' \
 *     -o ../src/wasm/gs_core.js
 *
 * The build artefacts (gs_core.js + gs_core.wasm) land directly in src/wasm/
 * and are imported by wasmBackend.ts at runtime via the Vite ?url import.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

extern "C" {

// ---------------------------------------------------------------------------
// Morton encoding helpers
// ---------------------------------------------------------------------------

static inline uint32_t expand_bits(uint32_t v) {
  v &= 0x3ffu;
  v = (v | (v << 16u)) & 0x030000ffu;
  v = (v | (v <<  8u)) & 0x0300f00fu;
  v = (v | (v <<  4u)) & 0x030c30c3u;
  v = (v | (v <<  2u)) & 0x030c30c3u;
  v = (v | (v <<  2u)) & 0x09249249u;
  return v;
}

static inline uint32_t morton3(uint32_t x, uint32_t y, uint32_t z) {
  return expand_bits(x) | (expand_bits(y) << 1u) | (expand_bits(z) << 2u);
}

void morton_codes(
    const float* centers,
    int          count,
    float        min_x, float min_y, float min_z,
    float        inv_x, float inv_y, float inv_z,
    uint32_t*    out_codes)
{
  for (int i = 0; i < count; ++i) {
    const int o = i * 3;
    const auto qx = static_cast<uint32_t>(
        std::clamp((centers[o]     - min_x) * inv_x * 1023.0f, 0.0f, 1023.0f));
    const auto qy = static_cast<uint32_t>(
        std::clamp((centers[o + 1] - min_y) * inv_y * 1023.0f, 0.0f, 1023.0f));
    const auto qz = static_cast<uint32_t>(
        std::clamp((centers[o + 2] - min_z) * inv_z * 1023.0f, 0.0f, 1023.0f));
    out_codes[i] = morton3(qx, qy, qz);
  }
}

// ---------------------------------------------------------------------------
// Radix sort — 4-pass, 8-bit buckets, O(n)
//
// Converts each float depth to a sortable uint32 by flipping the sign bit
// (and all bits for negatives) so the standard unsigned integer ordering
// matches ascending float order.  A classic 4-pass LSB radix sort then
// produces the final sorted-index array.
// ---------------------------------------------------------------------------

static inline uint32_t float_to_sortable(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  // Positive: flip MSB so 0.0 -> 0x80000000 (sits above all negatives).
  // Negative: flip all bits so -inf -> 0x00000000 (smallest uint value).
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

void sort_depth(
    const float* centers,
    int          count,
    const float* view_matrix,
    uint32_t*    out_indices)
{
  if (count <= 0) return;

  const float e2  = view_matrix[2];
  const float e6  = view_matrix[6];
  const float e10 = view_matrix[10];
  const float e14 = view_matrix[14];

  // Pre-compute all depths as sortable uint32 keys.
  std::vector<uint32_t> keys(count);
  for (int i = 0; i < count; ++i) {
    const int o = i * 3;
    const float z = e2  * centers[o]
                  + e6  * centers[o + 1]
                  + e10 * centers[o + 2]
                  + e14;
    keys[i] = float_to_sortable(z);
  }

  // Initialise index array 0..n-1.
  std::vector<uint32_t> indices(count);
  std::iota(indices.begin(), indices.end(), 0u);

  // 4-pass LSB radix sort (8 bits per pass -> 4 passes for 32 bits).
  std::vector<uint32_t> tmp_indices(count);
  std::vector<uint32_t> tmp_keys(count);

  for (int shift = 0; shift < 32; shift += 8) {
    uint32_t histogram[256] = {};

    for (int i = 0; i < count; ++i)
      ++histogram[(keys[i] >> shift) & 0xffu];

    // Exclusive prefix-sum
    uint32_t running = 0;
    for (auto& h : histogram) {
      const uint32_t c = h;
      h = running;
      running += c;
    }

    // Scatter
    for (int i = 0; i < count; ++i) {
      const uint32_t bucket = (keys[i] >> shift) & 0xffu;
      const uint32_t dst    = histogram[bucket]++;
      tmp_indices[dst]      = indices[i];
      tmp_keys[dst]         = keys[i];
    }

    std::swap(indices, tmp_indices);
    std::swap(keys,    tmp_keys);
  }

  // Copy result
  for (int i = 0; i < count; ++i)
    out_indices[i] = indices[i];
}

} // extern "C"
