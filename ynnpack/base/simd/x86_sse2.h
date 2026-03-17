// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_SSE2_H_

#include <cstdint>

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_sse2_base.h"  // IWYU pragma: export
#include "ynnpack/base/simd/x86_sse2_partial_load_store.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

using f32x8 = vec<float, 8>;
using s32x8 = vec<int32_t, 8>;
using s16x16 = vec<int16_t, 16>;
using bf16x16 = vec<bfloat16, 16>;
using f16x16 = vec<half, 16>;
using s8x32 = vec<int8_t, 32>;
using u8x32 = vec<uint8_t, 32>;
using f64x4 = vec<double, 4>;
using f32x16 = vec<float, 16>;

using s32x16 = vec<int32_t, 16>;

YNN_ALWAYS_INLINE f32x8 convert(bf16x8 b, float) {
  __m128i zero = _mm_setzero_si128();

  return {
      f32x4{_mm_castsi128_ps(_mm_unpacklo_epi16(zero, b.v))},
      f32x4{_mm_castsi128_ps(_mm_unpackhi_epi16(zero, b.v))},
  };
}

YNN_ALWAYS_INLINE s32x16 convert(s8x16 a, int32_t) {
  __m128i i8_lo = _mm_unpacklo_epi8(a.v, a.v);
  __m128i i8_hi = _mm_unpackhi_epi8(a.v, a.v);

  return {
      {s32x4{_mm_srai_epi32(_mm_unpacklo_epi16(i8_lo, i8_lo), 24)},
       s32x4{_mm_srai_epi32(_mm_unpackhi_epi16(i8_lo, i8_lo), 24)}},
      {s32x4{_mm_srai_epi32(_mm_unpacklo_epi16(i8_hi, i8_hi), 24)},
       s32x4{_mm_srai_epi32(_mm_unpackhi_epi16(i8_hi, i8_hi), 24)}},
  };
}

YNN_ALWAYS_INLINE s32x16 convert(u8x16 a, int32_t) {
  const __m128i zero = _mm_setzero_si128();
  __m128i i16_lo = _mm_unpacklo_epi8(a.v, zero);
  __m128i i16_hi = _mm_unpackhi_epi8(a.v, zero);

  return {
      {s32x4{_mm_unpacklo_epi16(i16_lo, zero)},
       s32x4{_mm_unpackhi_epi16(i16_lo, zero)}},
      {s32x4{_mm_unpacklo_epi16(i16_hi, zero)},
       s32x4{_mm_unpackhi_epi16(i16_hi, zero)}},
  };
}

YNN_ALWAYS_INLINE f32x4 convert(s32x4 x, float) {
  return f32x4{_mm_cvtepi32_ps(x.v)};
}

YNN_ALWAYS_INLINE s32x4 convert(f32x4 x, int32_t) {
  return s32x4{_mm_cvttps_epi32(x.v)};
}

YNN_ALWAYS_INLINE f64x4 convert(f32x4 x, double) {
  return {f64x2{_mm_cvtps_pd(x.v)},
          f64x2{_mm_cvtps_pd(_mm_movehl_ps(x.v, x.v))}};
}
YNN_ALWAYS_INLINE f32x4 convert(f64x4 x, float) {
  return f32x4{_mm_movelh_ps(_mm_cvtpd_ps(x[0].v), _mm_cvtpd_ps(x[1].v))};
}

YNN_ALWAYS_INLINE s16x8 saturating_convert(s32x8 a, int16_t) {
  return s16x8{_mm_packs_epi32(a.lo().v, a.hi().v)};
}

YNN_ALWAYS_INLINE s8x16 saturating_convert(s16x16 a, int8_t) {
  return s8x16{_mm_packs_epi16(a.lo().v, a.hi().v)};
}

YNN_ALWAYS_INLINE u8x16 saturating_convert(s16x16 a, uint8_t) {
  return u8x16{_mm_packus_epi16(a.lo().v, a.hi().v)};
}

YNN_ALWAYS_INLINE s16x8 saturating_rounding_convert(f32x8 f, int16_t) {
  const __m128 max_int16 = _mm_set1_ps((float)((1 << 15) - 1));
  const __m128i i0 = _mm_cvtps_epi32(_mm_min_ps(f.lo().v, max_int16));
  const __m128i i1 = _mm_cvtps_epi32(_mm_min_ps(f.hi().v, max_int16));
  return saturating_convert(s32x8(s32x4(i0), s32x4(i1)), int16_t());
}

YNN_ALWAYS_INLINE s8x16 saturating_rounding_convert(f32x16 f, int8_t) {
  const s16x8 i01 =
      saturating_rounding_convert(f32x8(f.lo().lo(), f.lo().hi()), int16_t());
  const s16x8 i23 =
      saturating_rounding_convert(f32x8(f.hi().lo(), f.hi().hi()), int16_t());
  return saturating_convert(s16x16(i01, i23), int8_t());
}

YNN_ALWAYS_INLINE u8x16 saturating_rounding_convert(f32x16 f, uint8_t) {
  const __m128 max_int16 = _mm_set1_ps((1 << 15) - 1);
  const __m128i i0 = _mm_cvtps_epi32(_mm_min_ps(f.lo().lo().v, max_int16));
  const __m128i i1 = _mm_cvtps_epi32(_mm_min_ps(f.lo().hi().v, max_int16));
  const __m128i i2 = _mm_cvtps_epi32(_mm_min_ps(f.hi().lo().v, max_int16));
  const __m128i i3 = _mm_cvtps_epi32(_mm_min_ps(f.hi().hi().v, max_int16));
  const __m128i i01_16 = _mm_packs_epi32(i0, i1);
  const __m128i i23_16 = _mm_packs_epi32(i2, i3);
  return u8x16{_mm_packs_epi16(i01_16, i23_16)};
}

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_SSE_H_
