#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

extern "C" {

static inline uint32_t expand_bits(uint32_t v) {
  v &= 0x3ffu;
  v = (v | (v << 16u)) & 0x030000ffu;
  v = (v | (v << 8u)) & 0x0300f00fu;
  v = (v | (v << 4u)) & 0x030c30c3u;
  v = (v | (v << 2u)) & 0x09249249u;
  return v;
}

uint32_t morton3(uint32_t x, uint32_t y, uint32_t z) {
  return expand_bits(x) | (expand_bits(y) << 1u) | (expand_bits(z) << 2u);
}

void morton_codes(
    const float* centers,
    int count,
    float min_x,
    float min_y,
    float min_z,
    float inv_x,
    float inv_y,
    float inv_z,
    uint32_t* out_codes) {
  for (int i = 0; i < count; ++i) {
    const int o = i * 3;
    const auto qx = static_cast<uint32_t>(std::clamp((centers[o] - min_x) * inv_x * 1023.0f, 0.0f, 1023.0f));
    const auto qy = static_cast<uint32_t>(std::clamp((centers[o + 1] - min_y) * inv_y * 1023.0f, 0.0f, 1023.0f));
    const auto qz = static_cast<uint32_t>(std::clamp((centers[o + 2] - min_z) * inv_z * 1023.0f, 0.0f, 1023.0f));
    out_codes[i] = morton3(qx, qy, qz);
  }
}

void sort_depth(const float* centers, int count, const float* view_matrix, uint32_t* out_indices) {
  std::vector<uint32_t> indices(count);
  std::iota(indices.begin(), indices.end(), 0u);

  std::sort(indices.begin(), indices.end(), [&](uint32_t a, uint32_t b) {
    const int ao = static_cast<int>(a) * 3;
    const int bo = static_cast<int>(b) * 3;
    const float az =
        view_matrix[2] * centers[ao] +
        view_matrix[6] * centers[ao + 1] +
        view_matrix[10] * centers[ao + 2] +
        view_matrix[14];
    const float bz =
        view_matrix[2] * centers[bo] +
        view_matrix[6] * centers[bo + 1] +
        view_matrix[10] * centers[bo + 2] +
        view_matrix[14];
    return az < bz;
  });

  std::copy(indices.begin(), indices.end(), out_indices);
}

}

