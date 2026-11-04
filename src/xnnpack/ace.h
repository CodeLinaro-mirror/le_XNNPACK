// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_ACE_H_
#define XNNPACK_SRC_XNNPACK_ACE_H_

#include <stdint.h>

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#endif
#endif

#include <immintrin.h>

#include "src/xnnpack/common.h"

#if XNN_ARCH_X86_64 && (defined(__clang__) || defined(__GNUC__))

static XNN_INLINE void xnn_ace_load_palette2_config(void) {
  // Palette 2 configures 16 architectural 16x16x32b tiles (tmm0..tmm15) with
  // fixed 16 rows x 64 bytes; rows[0..15] and colsb[0..15] must be zero.
  XNN_ALIGN(64) static const uint8_t tile_cfg[64] = {2};
  _tile_loadconfig(tile_cfg);
}

// ACEv1 TOP4BSSD tmmDst, zmm1, zmm2:
//   tmmDst[m, n] += sum_{c=0..3} s8(va[m, c]) * s8(vb[n, c])
// Encoded as EVEX.512.F2.0F38.W0 0x5E /r with ModRM.rm = zmm1 (va) and
// EVEX.vvvv = ~2 -> zmm2 (vb). Must be volatile because tmmDst is unmodeled
// architectural tile state.
#define XNN_DEFINE_ACE_TOP4BSSD(tmm_idx, modrm_hex)                    \
  static XNN_INLINE void xnn_ace_top4bssd_tmm##tmm_idx(__m512i va,     \
                                                       __m512i vb) {   \
    register __m512i z1 __asm__("zmm1") = va;                          \
    register __m512i z2 __asm__("zmm2") = vb;                          \
    __asm__ volatile(".byte 0x62, 0xf2, 0x6f, 0x48, 0x5e, " #modrm_hex \
                     :                                                 \
                     : "v"(z1), "v"(z2));                              \
  }

XNN_DEFINE_ACE_TOP4BSSD(0, 0xc1)
XNN_DEFINE_ACE_TOP4BSSD(1, 0xc9)
XNN_DEFINE_ACE_TOP4BSSD(2, 0xd1)
XNN_DEFINE_ACE_TOP4BSSD(3, 0xd9)

#undef XNN_DEFINE_ACE_TOP4BSSD

// AMX-AVX512 TILEMOVROW zmm0, tmmSrc, eax:
//   Extracts 64-byte row `row` (16 x int32_t) from tmmSrc directly into zmm0.
// Encoded as EVEX.512.66.0F38.W0 0x4A /r with ModRM.reg = zmm0, ModRM.rm =
// tmmSrc, and EVEX.vvvv = ~0 -> eax. Must be volatile because tmmSrc is
// unmodeled architectural tile state.
#define XNN_DEFINE_ACE_TILEMOVROW(tmm_idx, modrm_hex)                       \
  static XNN_INLINE __m512i xnn_ace_tilemovrow_tmm##tmm_idx(uint32_t row) { \
    register __m512i z0 __asm__("zmm0");                                    \
    __asm__ volatile(".byte 0x62, 0xf2, 0x7d, 0x48, 0x4a, " #modrm_hex      \
                     : "=v"(z0)                                             \
                     : "a"(row));                                           \
    __m512i out = z0;                                                       \
    XNN_MSAN_UNPOISON(&out, sizeof(out));                                   \
    return out;                                                             \
  }

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#define XNN_MSAN_UNPOISON(ptr, size) __msan_unpoison((ptr), (size))
#else
#define XNN_MSAN_UNPOISON(ptr, size) ((void)0)
#endif
#else
#define XNN_MSAN_UNPOISON(ptr, size) ((void)0)
#endif

XNN_DEFINE_ACE_TILEMOVROW(0, 0xc0)
XNN_DEFINE_ACE_TILEMOVROW(1, 0xc1)
XNN_DEFINE_ACE_TILEMOVROW(2, 0xc2)
XNN_DEFINE_ACE_TILEMOVROW(3, 0xc3)

#undef XNN_DEFINE_ACE_TILEMOVROW

// AVX10.2 VPDPBSSD zmmAcc, zmm1, zmm2:
//   zmmAcc[n] += sum_{c=0..3} s8(va[n, c]) * s8(vb[n, c])
// Encoded as EVEX.512.F2.0F38.W0 0x50 /r with EVEX.vvvv = ~1 -> zmm1 (va) and
// ModRM.rm = zmm2 (vb). Intentionally non-volatile because all inputs and
// outputs are modeled in operand constraints with no hidden side effects.
#define XNN_DEFINE_AVX10_DPBSSD(slot_idx, acc_reg, modrm_hex)   \
  static XNN_INLINE __m512i xnn_avx10_dpbssd_epi32_t##slot_idx( \
      __m512i vacc, __m512i va, __m512i vb) {                   \
    register __m512i zacc __asm__(acc_reg) = vacc;              \
    register __m512i z1 __asm__("zmm1") = va;                   \
    register __m512i z2 __asm__("zmm2") = vb;                   \
    __asm__(".byte 0x62, 0xf2, 0x77, 0x48, 0x50, " #modrm_hex   \
            : "+v"(zacc)                                        \
            : "v"(z1), "v"(z2));                                \
    __m512i out = zacc;                                         \
    XNN_MSAN_UNPOISON(&out, sizeof(out));                       \
    return out;                                                 \
  }

XNN_DEFINE_AVX10_DPBSSD(0, "zmm0", 0xc2)
XNN_DEFINE_AVX10_DPBSSD(1, "zmm3", 0xda)
XNN_DEFINE_AVX10_DPBSSD(2, "zmm4", 0xe2)
XNN_DEFINE_AVX10_DPBSSD(3, "zmm5", 0xea)

#undef XNN_DEFINE_AVX10_DPBSSD
#undef XNN_MSAN_UNPOISON

#elif XNN_ARCH_X86_64

static XNN_INLINE void xnn_ace_load_palette2_config(void) { XNN_UNREACHABLE; }

#define XNN_DEFINE_ACE_STUBS(idx)                                            \
  static XNN_INLINE void xnn_ace_top4bssd_tmm##idx(__m512i va, __m512i vb) { \
    (void)va;                                                                \
    (void)vb;                                                                \
    XNN_UNREACHABLE;                                                         \
  }                                                                          \
  static XNN_INLINE __m512i xnn_ace_tilemovrow_tmm##idx(uint32_t row) {      \
    (void)row;                                                               \
    XNN_UNREACHABLE;                                                         \
    return _mm512_setzero_si512();                                           \
  }                                                                          \
  static XNN_INLINE __m512i xnn_avx10_dpbssd_epi32_t##idx(                   \
      __m512i vacc, __m512i va, __m512i vb) {                                \
    (void)vacc;                                                              \
    (void)va;                                                                \
    (void)vb;                                                                \
    XNN_UNREACHABLE;                                                         \
    return _mm512_setzero_si512();                                           \
  }

XNN_DEFINE_ACE_STUBS(0)
XNN_DEFINE_ACE_STUBS(1)
XNN_DEFINE_ACE_STUBS(2)
XNN_DEFINE_ACE_STUBS(3)

#undef XNN_DEFINE_ACE_STUBS

#endif  // XNN_ARCH_X86_64

#endif  // XNNPACK_SRC_XNNPACK_ACE_H_
