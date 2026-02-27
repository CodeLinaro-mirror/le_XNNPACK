// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_DOT_X86_AMX_H_
#define XNNPACK_YNNPACK_KERNELS_DOT_X86_AMX_H_

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"

#if YNN_COMPILER_HAS_FEATURE(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#endif

#if defined(__GNUC__) && !defined(__clang__)
// Workaround for GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=122446
#define YNN_TILE_DP_IMPL(name, dst, src1, src2)                             \
  __asm__ volatile(                                                         \
      "{t" #name "\t%%tmm%c[_src2], %%tmm%c[_src1], %%tmm%c[_dst]|t" #name    \
      "\t%%tmm%c[_dst], %%tmm%c[_src1], %%tmm%c[_src1]}" ::[_dst] "i"(dst), \
      [_src1] "i"(src1), [_src2] "i"(src2))

#define YNN_TILE_DPBF16PS(a, b, c) YNN_TILE_DP_IMPL(dpbf16ps, a, b, c)
#define YNN_TILE_DPFP16PS(a, b, c) YNN_TILE_DP_IMPL(dpfp16ps, a, b, c)
#define YNN_TILE_DPBSSD(a, b, c) YNN_TILE_DP_IMPL(dpbssd, a, b, c)
#define YNN_TILE_DPBUSD(a, b, c) YNN_TILE_DP_IMPL(dpbusd, a, b, c)
#else
#define YNN_TILE_DPBF16PS(a, b, c) _tile_dpbf16ps(a, b, c)
#define YNN_TILE_DPFP16PS(a, b, c) _tile_dpfp16ps(a, b, c)
#define YNN_TILE_DPBSSD(a, b, c) _tile_dpbssd(a, b, c)
#define YNN_TILE_DPBUSD(a, b, c) _tile_dpbusd(a, b, c)
#endif

namespace ynn {

namespace internal {

constexpr size_t kTileRowBytes = 64;

struct tile_config {
  std::uint8_t palette_id;
  std::uint8_t start_row;
  std::uint8_t reserved_0[14];
  std::uint16_t colsb[8];
  std::uint16_t reserved_1[8];
  std::uint8_t rows[8];
  std::uint8_t reserved_2[8];
};

static_assert(sizeof(tile_config) == 64, "tile_config size mismatch");
static_assert(offsetof(tile_config, colsb) == 16,
              "tile_config colsb offset mismatch");
static_assert(offsetof(tile_config, rows) == 48,
              "tile_config rows offset mismatch");

template <typename TA, typename TB, typename TC>
static void load_tile_config_1x4(size_t m, size_t n, size_t ktail) {
  alignas(64) tile_config config = {0};
  config.palette_id = 1;

  // c tiles
  config.rows[0] = m;
  config.rows[1] = m;
  config.rows[2] = m;
  config.rows[3] = m;
  config.colsb[0] = n * sizeof(TC);
  config.colsb[1] = n * sizeof(TC);
  config.colsb[2] = n * sizeof(TC);
  config.colsb[3] = n * sizeof(TC);

  // a, a tail
  config.rows[4] = m;
  config.colsb[4] = 64;
  config.rows[6] = m;
  config.colsb[6] = ktail * sizeof(TA);

  // b, b tail
  config.rows[5] = 16;
  config.colsb[5] = n * sizeof(TC);
  config.rows[7] = ktail * sizeof(TB) / 4;
  config.colsb[7] = n * sizeof(TC);

  _tile_loadconfig(&config);
}

}  // namespace internal

template <typename TAB, typename TC, template <int, int, int> class TileOp>
YNN_ALWAYS_INLINE static void x86_amx_dot(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  // AMX is structured as 16x16x4 byte tiles. Each row is 64 bytes. This will
  // represent 64 / sizeof(T) elements.
  constexpr size_t k_block = internal::kTileRowBytes / sizeof(TAB);

  assert(M > 0);
  assert(N > 0);
  assert(K3 > 0);
  assert(K2 > 0);
  assert(K1 > 0);
  assert(M <= 16);

  constexpr size_t align_k = 4 / sizeof(TAB);
  assert(K1 % align_k == 0);

  const size_t B_stride_k1_block = B_stride_k1 * k_block;

  // We load this many rows of B at a time.
  B_stride_k1 *= align_k;
  assert(B_stride_k1 % internal::kTileRowBytes == 0 || K1 == 1);

  // The size of the remainder in the K loop.
  const size_t k_tail = (K1 & (k_block - 1)) ? (K1 & (k_block - 1)) : k_block;

  // Prepare the config for the main loop (4 tiles).
  internal::load_tile_config_1x4<TAB, TAB, TC>(M, 16, k_tail);
  while (N >= 64) {
    if (C_in) {
      _tile_loadd(0, offset_bytes(C_in, 0 * internal::kTileRowBytes),
                  C_in_stride_m);
      _tile_loadd(1, offset_bytes(C_in, 1 * internal::kTileRowBytes),
                  C_in_stride_m);
      _tile_loadd(2, offset_bytes(C_in, 2 * internal::kTileRowBytes),
                  C_in_stride_m);
      _tile_loadd(3, offset_bytes(C_in, 3 * internal::kTileRowBytes),
                  C_in_stride_m);
    } else {
      _tile_zero(0);
      _tile_zero(1);
      _tile_zero(2);
      _tile_zero(3);
    }
    const void* B_k3 = B;
    const void* A_k3 = A;
    size_t k3 = K3;
    do {
      const void* B_k2 = B_k3;
      const void* A_k2 = A_k3;
      size_t k2 = K2;
      do {
        const void* B_k1 = B_k2;
        const void* A_k1 = A_k2;
        std::ptrdiff_t k1 = K1;
        while (k1 >= k_block) {
          _tile_loadd(4, A_k1, A_stride_m);

          _tile_loadd(5, offset_bytes(B_k1, (0 * internal::kTileRowBytes)),
                      B_stride_k1);
          TileOp<0, 4, 5>()();
          _tile_loadd(5, offset_bytes(B_k1, (1 * internal::kTileRowBytes)),
                      B_stride_k1);
          TileOp<1, 4, 5>()();
          _tile_loadd(5, offset_bytes(B_k1, (2 * internal::kTileRowBytes)),
                      B_stride_k1);
          TileOp<2, 4, 5>()();
          _tile_loadd(5, offset_bytes(B_k1, (3 * internal::kTileRowBytes)),
                      B_stride_k1);
          TileOp<3, 4, 5>()();

          k1 -= k_block;
          B_k1 = offset_bytes(B_k1, B_stride_k1_block);
          A_k1 = offset_bytes(A_k1, internal::kTileRowBytes);
        }
        if (k1 > 0) {
          _tile_loadd(6, A_k1, A_stride_m);
          _tile_loadd(7, offset_bytes(B_k1, (0 * internal::kTileRowBytes)),
                      B_stride_k1);
          TileOp<0, 6, 7>()();
          _tile_loadd(7, offset_bytes(B_k1, (1 * internal::kTileRowBytes)),
                      B_stride_k1);
          TileOp<1, 6, 7>()();
          _tile_loadd(7, offset_bytes(B_k1, (2 * internal::kTileRowBytes)),
                      B_stride_k1);
          TileOp<2, 6, 7>()();
          _tile_loadd(7, offset_bytes(B_k1, (3 * internal::kTileRowBytes)),
                      B_stride_k1);
          TileOp<3, 6, 7>()();
        }
        k2 -= 1;
        B_k2 = offset_bytes(B_k2, B_stride_k2);
        A_k2 = offset_bytes(A_k2, A_stride_k2);
      } while (k2 > 0);
      k3 -= 1;
      B_k3 = offset_bytes(B_k3, B_stride_k3);
      A_k3 = offset_bytes(A_k3, A_stride_k3);
    } while (k3 > 0);
    _tile_stored(0, offset_bytes(C_out, 0 * internal::kTileRowBytes),
                 C_out_stride_m);
    _tile_stored(1, offset_bytes(C_out, 1 * internal::kTileRowBytes),
                 C_out_stride_m);
    _tile_stored(2, offset_bytes(C_out, 2 * internal::kTileRowBytes),
                 C_out_stride_m);
    _tile_stored(3, offset_bytes(C_out, 3 * internal::kTileRowBytes),
                 C_out_stride_m);
#if YNN_COMPILER_HAS_FEATURE(memory_sanitizer)
    // msan doesn't support amx, avoid false positives.
    for (size_t i = 0; i < M; ++i) {
      __msan_unpoison(offset_bytes(C_out, i * C_out_stride_m),
                      4 * internal::kAmxTileRowBytes);
    }
#endif
    C_in = C_in ? offset_bytes(C_in, 4 * internal::kTileRowBytes) : nullptr;
    C_out = offset_bytes(C_out, 4 * internal::kTileRowBytes);
    B = offset_bytes(B, 4 * internal::kTileRowBytes);
    N -= 64;
  }
  while (N > 0) {
    // We might need to handle a less-than-tile here.
    internal::load_tile_config_1x4<TAB, TAB, TC>(M, std::min<size_t>(N, 16),
                                                 k_tail);
    if (C_in) {
      _tile_loadd(0, C_in, C_in_stride_m);
    } else {
      _tile_zero(0);
    }
    const void* B_k3 = B;
    const void* A_k3 = A;
    size_t k3 = K3;
    do {
      const void* B_k2 = B_k3;
      const void* A_k2 = A_k3;
      size_t k2 = K2;
      do {
        const void* B_k1 = B_k2;
        const void* A_k1 = A_k2;
        std::ptrdiff_t k1 = K1;
        while (k1 >= k_block) {
          _tile_loadd(4, A_k1, A_stride_m);
          _tile_loadd(5, B_k1, B_stride_k1);
          TileOp<0, 4, 5>()();

          k1 -= k_block;
          B_k1 = offset_bytes(B_k1, B_stride_k1_block);
          A_k1 = offset_bytes(A_k1, internal::kTileRowBytes);
        }
        if (k1 > 0) {
          _tile_loadd(6, A_k1, A_stride_m);
          _tile_loadd(7, B_k1, B_stride_k1);
          TileOp<0, 6, 7>()();
        }
        k2 -= 1;
        B_k2 = offset_bytes(B_k2, B_stride_k2);
        A_k2 = offset_bytes(A_k2, A_stride_k2);
      } while (k2 > 0);
      k3 -= 1;
      B_k3 = offset_bytes(B_k3, B_stride_k3);
      A_k3 = offset_bytes(A_k3, A_stride_k3);
    } while (k3 > 0);
    _tile_stored(0, C_out, C_out_stride_m);
    #if YNN_COMPILER_HAS_FEATURE(memory_sanitizer)
    // msan doesn't support amx, avoid false positives.
    for (size_t i = 0; i < M; ++i) {
      __msan_unpoison(offset_bytes(C_out, i * C_out_stride_m), N * sizeof(TC));
    }
    #endif
    C_in = C_in ? offset_bytes(C_in, internal::kTileRowBytes) : nullptr;
    C_out = offset_bytes(C_out, internal::kTileRowBytes);
    B = offset_bytes(B, internal::kTileRowBytes);
    N = sub_sat(N, 16);
  }
  _tile_release();
}

namespace internal {

// Loads the tile configuration for a 2x2 scenario with the given dimensions.
//
// Parameters:
//   m: number of rows in A and C.
//   n: number of columns within a tile of B and C.
//   k_len: length of the K dimension (must be <= 16).
//   align_k: alignment of the K dimension in the B matrix.
template <typename TAB, typename TC>
static void load_config_2x2(size_t m, size_t n, size_t k_len, size_t align_k) {
  const size_t m0 = std::min<size_t>(m, 16);
  const size_t m1 = m > 16 ? std::min<size_t>(m - 16, 16) : 0;

  const size_t n0 = std::min<size_t>(n, 16);
  const size_t n1 = n > 16 ? std::min<size_t>(n - 16, 16) : 0;

  assert(m0 > 0);
  assert(n > 0);
  assert(n <= 32);
  assert(k_len > 0);
  assert(m0 <= 16);
  assert(m1 <= 16);
  assert(n0 <= 16);
  assert(n1 <= 16);

  alignas(64) tile_config config = {0};
  config.palette_id = 1;

  // C tiles.
  config.rows[0] = m0;
  config.colsb[0] = n0 * sizeof(TC);
  if (n1 > 0) {
    config.rows[1] = m0;
    config.colsb[1] = n1 * sizeof(TC);
  }

  if (m1 > 0) {
    config.rows[2] = m1;
    config.colsb[2] = n0 * sizeof(TC);
    if (n1 > 0) {
      config.rows[3] = m1;
      config.colsb[3] = n1 * sizeof(TC);
    }
  }

  // A tiles.
  config.rows[4] = m0;
  config.colsb[4] = k_len * sizeof(TAB);
  if (m1 > 0) {
    config.rows[6] = m1;
    config.colsb[6] = k_len * sizeof(TAB);
  }

  // B tiles.
  config.rows[5] = k_len / align_k;
  config.colsb[5] = n0 * sizeof(TC);
  if (n1 > 0) {
    config.rows[7] = k_len / align_k;
    config.colsb[7] = n1 * sizeof(TC);
  }

  _tile_loadconfig(&config);
}

template <template <int, int, int> class TileOp, bool m0_n0, bool m0_n1,
          bool m1_n0, bool m1_n1>
struct ComputeSequenceImpl {
  YNN_ALWAYS_INLINE static void Run(const void* A, size_t A_stride,
                                    const void* B, size_t B_stride);
};

// Case 1: 1x1 (m0_n0)
template <template <int, int, int> class TileOp>
struct ComputeSequenceImpl<TileOp, /*m0_n0=*/true, /*m0_n1=*/false,
                           /*m1_n0=*/false, /*m1_n1=*/false> {
  YNN_ALWAYS_INLINE static void Run(const void* A, size_t A_stride,
                                    const void* B, size_t B_stride) {
    _tile_loadd(4, A, A_stride);
    _tile_loadd(5, B, B_stride);
    TileOp<0, 4, 5>()();
  }
};

// Case 2: 1x2 (m0_n0, m0_n1)
template <template <int, int, int> class TileOp>
struct ComputeSequenceImpl<TileOp, /*m0_n0=*/true, /*m0_n1=*/true,
                           /*m1_n0=*/false, /*m1_n1=*/false> {
  YNN_ALWAYS_INLINE static void Run(const void* A, size_t A_stride,
                                    const void* B, size_t B_stride) {
    _tile_loadd(4, A, A_stride);
    _tile_loadd(5, offset_bytes(B, 0), B_stride);
    _tile_loadd(7, offset_bytes(B, kTileRowBytes), B_stride);
    TileOp<0, 4, 5>()();
    TileOp<1, 4, 7>()();
  }
};

// Case 3: 2x1 (m0_n0, m1_n0)
template <template <int, int, int> class TileOp>
struct ComputeSequenceImpl<TileOp, /*m0_n0=*/true, /*m0_n1=*/false,
                           /*m1_n0=*/true, /*m1_n1=*/false> {
  YNN_ALWAYS_INLINE static void Run(const void* A, size_t A_stride,
                                    const void* B, size_t B_stride) {
    _tile_loadd(4, A, A_stride);
    _tile_loadd(5, B, B_stride);
    _tile_loadd(6, offset_bytes(A, 16 * A_stride), A_stride);
    TileOp<0, 4, 5>()();
    TileOp<2, 6, 5>()();
  }
};

// Case 4: 2x2 (m0_n0, m0_n1, m1_n0, m1_n1)
template <template <int, int, int> class TileOp>
struct ComputeSequenceImpl<TileOp, /*m0_n0=*/true, /*m0_n1=*/true,
                           /*m1_n0=*/true, /*m1_n1=*/true> {
  YNN_ALWAYS_INLINE static void Run(const void* A, size_t A_stride,
                                    const void* B, size_t B_stride) {
    _tile_loadd(4, A, A_stride);
    _tile_loadd(5, offset_bytes(B, 0), B_stride);
    _tile_loadd(6, offset_bytes(A, 16 * A_stride), A_stride);
    TileOp<0, 4, 5>()();
    _tile_loadd(7, offset_bytes(B, kTileRowBytes), B_stride);
    TileOp<2, 6, 5>()();
    TileOp<1, 4, 7>()();
    TileOp<3, 6, 7>()();
  }
};

template <template <int, int, int> class TileOp, bool m0_n0, bool m0_n1,
          bool m1_n0, bool m1_n1>
YNN_ALWAYS_INLINE static void ComputeSequence(const void* A, size_t A_stride,
                                              const void* B, size_t B_stride) {
  ComputeSequenceImpl<TileOp, m0_n0, m0_n1, m1_n0, m1_n1>::Run(A, A_stride, B,
                                                               B_stride);
}

template <template <int, int, int> class TileOp, bool m0_n0, bool m0_n1,
          bool m1_n0, bool m1_n1>
YNN_ALWAYS_INLINE static void ComputeOverK(
    size_t K3, size_t K2, size_t k1_loops, const void* A, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, size_t A_k1_increment,
    const void* B, size_t B_stride_k3, size_t B_stride_k2, size_t B_stride_k1,
    size_t B_k1_increment) {
  const void* B_k3 = B;
  const void* A_k3 = A;
  size_t k3 = K3;
  do {
    const void* B_k2 = B_k3;
    const void* A_k2 = A_k3;
    size_t k2 = K2;
    do {
      const void* B_k1 = B_k2;
      const void* A_k1 = A_k2;
      size_t k = k1_loops;
      while (k--) {
        ComputeSequence<TileOp, m0_n0, m0_n1, m1_n0, m1_n1>(A_k1, A_stride_m,
                                                            B_k1, B_stride_k1);
        B_k1 = offset_bytes(B_k1, B_k1_increment);
        A_k1 = offset_bytes(A_k1, A_k1_increment);
      }
      k2 -= 1;
      B_k2 = offset_bytes(B_k2, B_stride_k2);
      A_k2 = offset_bytes(A_k2, A_stride_k2);
    } while (k2 > 0);
    k3 -= 1;
    B_k3 = offset_bytes(B_k3, B_stride_k3);
    A_k3 = offset_bytes(A_k3, A_stride_k3);
  } while (k3 > 0);
}

template <template <int, int, int> class TileOp, bool m0_n0, bool m0_n1,
          bool m1_n0, bool m1_n1>
struct ComputeBlockImpl {
  YNN_ALWAYS_INLINE static void Run(size_t K3, size_t K2, size_t k1_loops,
                                    const void* A, size_t A_stride_m,
                                    size_t A_stride_k3, size_t A_stride_k2,
                                    size_t A_k1_increment, const void* B,
                                    size_t B_stride_k3, size_t B_stride_k2,
                                    size_t B_stride_k1, size_t B_k1_increment,
                                    const void* C_in, size_t C_in_stride,
                                    void* C_out, size_t C_out_stride);
};

template <template <int, int, int> class TileOp>
struct ComputeBlockImpl<TileOp, /*m0_n0=*/true, /*m0_n1=*/false,
                        /*m1_n0=*/false, /*m1_n1=*/false> {
  YNN_ALWAYS_INLINE static void Run(size_t K3, size_t K2, size_t k1_loops,
                                    const void* A, size_t A_stride_m,
                                    size_t A_stride_k3, size_t A_stride_k2,
                                    size_t A_k1_increment, const void* B,
                                    size_t B_stride_k3, size_t B_stride_k2,
                                    size_t B_stride_k1, size_t B_k1_increment,
                                    const void* C_in, size_t C_in_stride,
                                    void* C_out, size_t C_out_stride) {
    if (C_in) {
      _tile_loadd(0, offset_bytes(C_in, 0), C_in_stride);
    } else {
      _tile_zero(0);
    }

    ComputeOverK<TileOp, /*m0_n0=*/true, /*m0_n1=*/false, /*m1_n0=*/false,
                 /*m1_n1=*/false>(K3, K2, k1_loops, A, A_stride_m, A_stride_k3,
                                  A_stride_k2, A_k1_increment, B, B_stride_k3,
                                  B_stride_k2, B_stride_k1, B_k1_increment);

    _tile_stored(0, offset_bytes(C_out, 0), C_out_stride);
  }
};

template <template <int, int, int> class TileOp>
struct ComputeBlockImpl<TileOp, /*m0_n0=*/true, /*m0_n1=*/true, /*m1_n0=*/false,
                        /*m1_n1=*/false> {
  YNN_ALWAYS_INLINE static void Run(size_t K3, size_t K2, size_t k1_loops,
                                    const void* A, size_t A_stride_m,
                                    size_t A_stride_k3, size_t A_stride_k2,
                                    size_t A_k1_increment, const void* B,
                                    size_t B_stride_k3, size_t B_stride_k2,
                                    size_t B_stride_k1, size_t B_k1_increment,
                                    const void* C_in, size_t C_in_stride,
                                    void* C_out, size_t C_out_stride) {
    if (C_in) {
      _tile_loadd(0, offset_bytes(C_in, 0), C_in_stride);
      _tile_loadd(1, offset_bytes(C_in, kTileRowBytes), C_in_stride);
    } else {
      _tile_zero(0);
      _tile_zero(1);
    }

    ComputeOverK<TileOp, /*m0_n0=*/true, /*m0_n1=*/true, /*m1_n0=*/false,
                 /*m1_n1=*/false>(K3, K2, k1_loops, A, A_stride_m, A_stride_k3,
                                  A_stride_k2, A_k1_increment, B, B_stride_k3,
                                  B_stride_k2, B_stride_k1, B_k1_increment);

    _tile_stored(0, offset_bytes(C_out, 0), C_out_stride);
    _tile_stored(1, offset_bytes(C_out, kTileRowBytes), C_out_stride);
  }
};

template <template <int, int, int> class TileOp>
struct ComputeBlockImpl<TileOp, /*m0_n0=*/true, /*m0_n1=*/false,
                        /*m1_n0=*/true, /*m1_n1=*/false> {
  YNN_ALWAYS_INLINE static void Run(size_t K3, size_t K2, size_t k1_loops,
                                    const void* A, size_t A_stride_m,
                                    size_t A_stride_k3, size_t A_stride_k2,
                                    size_t A_k1_increment, const void* B,
                                    size_t B_stride_k3, size_t B_stride_k2,
                                    size_t B_stride_k1, size_t B_k1_increment,
                                    const void* C_in, size_t C_in_stride,
                                    void* C_out, size_t C_out_stride) {
    if (C_in) {
      _tile_loadd(0, offset_bytes(C_in, 0), C_in_stride);
      size_t row_offset = 16 * C_in_stride;
      _tile_loadd(2, offset_bytes(C_in, row_offset), C_in_stride);
    } else {
      _tile_zero(0);
      _tile_zero(2);
    }

    ComputeOverK<TileOp, /*m0_n0=*/true, /*m0_n1=*/false, /*m1_n0=*/true,
                 /*m1_n1=*/false>(K3, K2, k1_loops, A, A_stride_m, A_stride_k3,
                                  A_stride_k2, A_k1_increment, B, B_stride_k3,
                                  B_stride_k2, B_stride_k1, B_k1_increment);

    _tile_stored(0, offset_bytes(C_out, 0), C_out_stride);
    size_t row_offset = 16 * C_out_stride;
    _tile_stored(2, offset_bytes(C_out, row_offset), C_out_stride);
  }
};

template <template <int, int, int> class TileOp>
struct ComputeBlockImpl<TileOp, /*m0_n0=*/true, /*m0_n1=*/true, /*m1_n0=*/true,
                        /*m1_n1=*/true> {
  YNN_ALWAYS_INLINE static void Run(size_t K3, size_t K2, size_t k1_loops,
                                    const void* A, size_t A_stride_m,
                                    size_t A_stride_k3, size_t A_stride_k2,
                                    size_t A_k1_increment, const void* B,
                                    size_t B_stride_k3, size_t B_stride_k2,
                                    size_t B_stride_k1, size_t B_k1_increment,
                                    const void* C_in, size_t C_in_stride,
                                    void* C_out, size_t C_out_stride) {
    if (C_in) {
      _tile_loadd(0, offset_bytes(C_in, 0), C_in_stride);
      _tile_loadd(1, offset_bytes(C_in, kTileRowBytes), C_in_stride);
      size_t row_offset = 16 * C_in_stride;
      _tile_loadd(2, offset_bytes(C_in, row_offset), C_in_stride);
      _tile_loadd(3, offset_bytes(C_in, row_offset + kTileRowBytes),
                  C_in_stride);
    } else {
      _tile_zero(0);
      _tile_zero(1);
      _tile_zero(2);
      _tile_zero(3);
    }

    ComputeOverK<TileOp, /*m0_n0=*/true, /*m0_n1=*/true, /*m1_n0=*/true,
                 /*m1_n1=*/true>(K3, K2, k1_loops, A, A_stride_m, A_stride_k3,
                                 A_stride_k2, A_k1_increment, B, B_stride_k3,
                                 B_stride_k2, B_stride_k1, B_k1_increment);

    _tile_stored(0, offset_bytes(C_out, 0), C_out_stride);
    _tile_stored(1, offset_bytes(C_out, kTileRowBytes), C_out_stride);
    size_t row_offset = 16 * C_out_stride;
    _tile_stored(2, offset_bytes(C_out, row_offset), C_out_stride);
    _tile_stored(3, offset_bytes(C_out, row_offset + kTileRowBytes),
                 C_out_stride);
  }
};

template <template <int, int, int> class TileOp, bool m0_n0, bool m0_n1,
          bool m1_n0, bool m1_n1>
YNN_ALWAYS_INLINE static void ComputeBlock(
    size_t K3, size_t K2, size_t k1_loops, const void* A, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, size_t A_k1_increment,
    const void* B, size_t B_stride_k3, size_t B_stride_k2, size_t B_stride_k1,
    size_t B_k1_increment, const void* C_in, size_t C_in_stride, void* C_out,
    size_t C_out_stride) {
  ComputeBlockImpl<TileOp, m0_n0, m0_n1, m1_n0, m1_n1>::Run(
      K3, K2, k1_loops, A, A_stride_m, A_stride_k3, A_stride_k2, A_k1_increment,
      B, B_stride_k3, B_stride_k2, B_stride_k1, B_k1_increment, C_in,
      C_in_stride, C_out, C_out_stride);
}

template <typename TAB, typename TC, template <int, int, int> class TileOp,
          bool has_second_row>
YNN_ALWAYS_INLINE static void x86_amx_dot_2x2_impl(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  constexpr size_t k_block = kTileRowBytes / sizeof(TAB);
  constexpr size_t align_k = 4 / sizeof(TAB);

  assert(M > 0);
  assert(N > 0);
  assert(K3 > 0);
  assert(K2 > 0);
  assert(K1 > 0);
  assert(M <= 32);

  const size_t B_stride_k1_block = B_stride_k1 * k_block;

  // We load this many rows of B at a time.
  B_stride_k1 *= align_k;
  assert(K1 % align_k == 0);

  const size_t k_tail = K1 & (k_block - 1);
  const size_t k1_iters = K1 / k_block;
  const size_t n_loops = N / 32;
  const size_t n_tail = N % 32;

  // 1. Handle M == 32 (or M == 16 if has_second_row == false), N <= 32,
  //    K == 32.
  if (k1_iters > 0) {
    load_config_2x2<TAB, TC>(M, 32, k_block, align_k);

    const void* current_C_in = C_in;
    void* current_C_out = C_out;
    const void* current_B = B;

    for (size_t i = 0; i < n_loops; ++i) {
      ComputeBlock<TileOp, /*m0_n0=*/true, /*m0_n1=*/true,
                   /*m1_n0=*/has_second_row, /*m1_n1=*/has_second_row>(
          K3, K2, k1_iters, A, A_stride_m, A_stride_k3, A_stride_k2,
          kTileRowBytes, current_B, B_stride_k3, B_stride_k2, B_stride_k1,
          B_stride_k1_block, current_C_in, C_in_stride_m, current_C_out,
          C_out_stride_m);

      if (current_C_in)
        current_C_in = offset_bytes(current_C_in, 2 * kTileRowBytes);
      current_C_out = offset_bytes(current_C_out, 2 * kTileRowBytes);
      current_B = offset_bytes(current_B, 2 * kTileRowBytes);
    }

    if (n_tail > 0) {
      load_config_2x2<TAB, TC>(M, n_tail, k_block, align_k);
      if (n_tail > 16) {
        ComputeBlock<TileOp, /*m0_n0=*/true, /*m0_n1=*/true,
                     /*m1_n0=*/has_second_row, /*m1_n1=*/has_second_row>(
            K3, K2, k1_iters, A, A_stride_m, A_stride_k3, A_stride_k2,
            kTileRowBytes, current_B, B_stride_k3, B_stride_k2, B_stride_k1,
            B_stride_k1_block, current_C_in, C_in_stride_m, current_C_out,
            C_out_stride_m);
      } else {
        ComputeBlock<TileOp, /*m0_n0=*/true, /*m0_n1=*/false,
                     /*m1_n0=*/has_second_row, /*m1_n1=*/false>(
            K3, K2, k1_iters, A, A_stride_m, A_stride_k3, A_stride_k2,
            kTileRowBytes, current_B, B_stride_k3, B_stride_k2, B_stride_k1,
            B_stride_k1_block, current_C_in, C_in_stride_m, current_C_out,
            C_out_stride_m);
      }
    }
  }

  // 2. Handle M == 32 (or M == 16 if has_second_row == false), N <= 32, K < 32.
  if (k_tail > 0) {
    load_config_2x2<TAB, TC>(M, 32, k_tail, align_k);

    // If k1_iters > 0, we load the accumulator value from C_out instead of the
    // initial value from C_in.
    const void* current_C_in = (k1_iters > 0) ? C_out : C_in;
    void* current_C_out = C_out;
    const void* current_B = B;

    const size_t k_main_offset_bytes = k1_iters * kTileRowBytes;
    const size_t k_main_offset_B = k1_iters * B_stride_k1_block;

    size_t stride =
        (current_C_in == current_C_out) ? C_out_stride_m : C_in_stride_m;

    for (size_t i = 0; i < n_loops; ++i) {
      ComputeBlock<TileOp, /*m0_n0=*/true, /*m0_n1=*/true,
                   /*m1_n0=*/has_second_row, /*m1_n1=*/has_second_row>(
          K3, K2, /*k1_loops=*/1, offset_bytes(A, k_main_offset_bytes),
          A_stride_m, A_stride_k3, A_stride_k2, 0,
          offset_bytes(current_B, k_main_offset_B), B_stride_k3, B_stride_k2,
          B_stride_k1, 0, current_C_in, stride, current_C_out, C_out_stride_m);

      if (current_C_in)
        current_C_in = offset_bytes(current_C_in, 2 * kTileRowBytes);
      current_C_out = offset_bytes(current_C_out, 2 * kTileRowBytes);
      current_B = offset_bytes(current_B, 2 * kTileRowBytes);
    }

    if (n_tail > 0) {
      load_config_2x2<TAB, TC>(M, n_tail, k_tail, align_k);
      if (n_tail > 16) {
        ComputeBlock<TileOp, /*m0_n0=*/true, /*m0_n1=*/true,
                     /*m1_n0=*/has_second_row, /*m1_n1=*/has_second_row>(
            K3, K2, /*k1_loops=*/1, offset_bytes(A, k_main_offset_bytes),
            A_stride_m, A_stride_k3, A_stride_k2, 0, current_B, B_stride_k3,
            B_stride_k2, B_stride_k1, 0, current_C_in, stride, current_C_out,
            C_out_stride_m);
      } else {
        ComputeBlock<TileOp, /*m0_n0=*/true, /*m0_n1=*/false,
                     /*m1_n0=*/has_second_row, /*m1_n1=*/false>(
            K3, K2, /*k1_loops=*/1, offset_bytes(A, k_main_offset_bytes),
            A_stride_m, A_stride_k3, A_stride_k2, 0, current_B, B_stride_k3,
            B_stride_k2, B_stride_k1, 0, current_C_in, stride, current_C_out,
            C_out_stride_m);
      }
    }
  }

  _tile_release();
}
}  // namespace internal

template <typename TAB, typename TC, template <int, int, int> class TileOp>
YNN_ALWAYS_INLINE static void x86_amx_dot_2x2(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  if (M > 16) {
    internal::x86_amx_dot_2x2_impl<TAB, TC, TileOp, /*has_second_row=*/true>(
        M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2, A, B_stride_k3,
        B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in, C_out_stride_m,
        C_out);
  } else {
    internal::x86_amx_dot_2x2_impl<TAB, TC, TileOp, /*has_second_row=*/false>(
        M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2, A, B_stride_k3,
        B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in, C_out_stride_m,
        C_out);
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DOT_X86_AMX_H_
