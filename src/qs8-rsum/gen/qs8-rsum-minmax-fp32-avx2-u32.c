// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/reduce.h>

void xnn_qs8_rsum_minmax_fp32_ukernel__avx2_u32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  const __m256i vone = _mm256_set1_epi8(INT8_C(1));
  const __m256i vone_16 = _mm256_set1_epi16(INT16_C(1));
  __m256i vacc0 = _mm256_setzero_si256();


  if (XNN_UNLIKELY(batch != 0)) {
    __m256i vacc16_0 = _mm256_setzero_si256();
    for (; batch >= 32; batch -= 32) {
      const __m256i vt0 = _mm256_maddubs_epi16(vone, _mm256_loadu_si256((const __m256i*) input)); input += 32;

      vacc16_0 = _mm256_add_epi16(vacc16_0, vt0);
    }

    for (; batch >= 32; batch -= 32) {
      const __m256i vt = _mm256_maddubs_epi16(vone, _mm256_loadu_si256((const __m256i*) input)); input += 32;
      vacc16_0 = _mm256_add_epi16(vacc16_0, vt);
    }
    if (XNN_UNLIKELY(batch != 0)) {
      assert(batch >= 1 && batch <= 31);
      const __m256i vonemask = _mm256_loadu_si256((const __m256i*) &params->fp32_avx2.mask_table[31 - batch]);
      const __m256i vt = _mm256_maddubs_epi16(vonemask, _mm256_loadu_si256((const __m256i*) input)); input += 32;
      vacc16_0 = _mm256_add_epi16(vacc16_0, vt);
    }

    vacc0 = _mm256_add_epi32(vacc0, _mm256_madd_epi16(vone_16, vacc16_0));
  }


  __m128i vacc_lo = _mm_add_epi32(_mm256_castsi256_si128(vacc0), _mm256_extractf128_si256(vacc0, 1));
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);
  const int32_t vacc = _mm_cvtsi128_si32(vacc_lo);

  const int32_t vinit_bias = params->fp32_avx2.init_bias;
  const float vscale = params->fp32_avx2.scale;
  const int32_t output_min = params->fp32_avx2.output_min;
  const int32_t output_max = params->fp32_avx2.output_max;
  const float vmagic_bias = params->fp32_avx2.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_avx2.magic_bias_less_output_zero_point;

  float vfpacc = (float) (vacc + vinit_bias) * vscale;
  vfpacc += vmagic_bias;
  int32_t vout = (int32_t) float_as_uint32(vfpacc);
  vout -= vmagic_bias_less_output_zero_point;
  vout = math_max_s32(vout, output_min);
  vout = math_min_s32(vout, output_max);
  *output += (int8_t) vout;
}
