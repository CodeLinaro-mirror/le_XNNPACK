// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx4c8-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>
#include <xnnpack/unaligned.h>


void xnn_qd8_f32_qs8w_gemm_minmax_ukernel_2x4c8__sse2_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0] * quantization_params[0].zero_point);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1] * quantization_params[0].zero_point);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2] * quantization_params[0].zero_point);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3] * quantization_params[0].zero_point);
    __m128i vacc1x0 = _mm_cvtsi32_si128(((const int*) w)[0] * quantization_params[1].zero_point);
    __m128i vacc1x1 = _mm_cvtsi32_si128(((const int*) w)[1] * quantization_params[1].zero_point);
    __m128i vacc1x2 = _mm_cvtsi32_si128(((const int*) w)[2] * quantization_params[1].zero_point);
    __m128i vacc1x3 = _mm_cvtsi32_si128(((const int*) w)[3] * quantization_params[1].zero_point);
    w = (const int32_t*) w + 4;

    size_t k = 0;
    while (k < kc) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_srai_epi16(_mm_unpacklo_epi8(va1, va1), 8);
      a1 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
      const __m128i vxb0 = _mm_srai_epi16(_mm_unpacklo_epi8(vb0, vb0), 8);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
      const __m128i vxb1 = _mm_srai_epi16(_mm_unpacklo_epi8(vb1, vb1), 8);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vxb2 = _mm_srai_epi16(_mm_unpacklo_epi8(vb2, vb2), 8);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
      const __m128i vxb3 = _mm_srai_epi16(_mm_unpacklo_epi8(vb3, vb3), 8);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));

      w = (const void*) ((const int8_t*) w + 32);
      k += 8 * sizeof(int8_t);
    }

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));
    const __m128i vacc1x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x0, vacc1x2), _mm_unpackhi_epi32(vacc1x0, vacc1x2));
    const __m128i vacc1x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x1, vacc1x3), _mm_unpackhi_epi32(vacc1x1, vacc1x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));
    __m128i vacc1x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x02, vacc1x13), _mm_unpackhi_epi32(vacc1x02, vacc1x13));

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vout1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    const __m128 vscale0 = _mm_set1_ps(quantization_params[0].scale);
    vout0x0123 = _mm_mul_ps(vout0x0123, vscale0);
    const __m128 vscale1 = _mm_set1_ps(quantization_params[1].scale);
    vout1x0123 = _mm_mul_ps(vout1x0123, vscale1);

    const __m128 vbias = _mm_load_ps(w); w = (const float*) w + 4;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias);
    vout1x0123 = _mm_add_ps(vout1x0123, vbias);
    const __m128 vmax = _mm_load_ps(&params->sse.max[0]);
    const __m128 vmin = _mm_load_ps(&params->sse.min[0]);
    vout0x0123 = _mm_max_ps(vout0x0123, vmin);
    vout0x0123 = _mm_min_ps(vout0x0123, vmax);
    vout1x0123 = _mm_max_ps(vout1x0123, vmin);
    vout1x0123 = _mm_min_ps(vout1x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c1, vout1x0123);
      _mm_storeu_ps(c0, vout0x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
        _mm_storel_pi((__m64*) c1, vout1x0123);
        vout1x0123 = _mm_unpackhi_ps(vout1x0123, vout1x0123);
        c1 += 2;
      }
      if (nc & 1) {
        unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(_mm_castps_si128(vout0x0123)));
        unaligned_store_u32(c1, (uint32_t) _mm_cvtsi128_si32(_mm_castps_si128(vout1x0123)));
      }
      nc = 0;
    }
  } while (nc != 0);
}
