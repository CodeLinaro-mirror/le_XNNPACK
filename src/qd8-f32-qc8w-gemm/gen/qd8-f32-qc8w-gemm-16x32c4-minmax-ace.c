// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-ace.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/ace.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/unaligned.h"


void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x32c4__ace(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params)
{
  assert(mr != 0);
  assert(mr <= 16);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

#if XNN_ARCH_X86_64
  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);

  const int8_t* a0 = a;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
  }
  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
  }
  const int8_t* a7 = (const int8_t*) ((uintptr_t) a6 + a_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    a7 = a6;
  }
  const int8_t* a8 = (const int8_t*) ((uintptr_t) a7 + a_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    a8 = a7;
  }
  const int8_t* a9 = (const int8_t*) ((uintptr_t) a8 + a_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    a9 = a8;
  }
  const int8_t* a10 = (const int8_t*) ((uintptr_t) a9 + a_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    a10 = a9;
  }
  const int8_t* a11 = (const int8_t*) ((uintptr_t) a10 + a_stride);
  if XNN_UNPREDICTABLE(mr < 12) {
    a11 = a10;
  }
  const int8_t* a12 = (const int8_t*) ((uintptr_t) a11 + a_stride);
  if XNN_UNPREDICTABLE(mr <= 12) {
    a12 = a11;
  }
  const int8_t* a13 = (const int8_t*) ((uintptr_t) a12 + a_stride);
  if XNN_UNPREDICTABLE(mr < 14) {
    a13 = a12;
  }
  const int8_t* a14 = (const int8_t*) ((uintptr_t) a13 + a_stride);
  if XNN_UNPREDICTABLE(mr <= 14) {
    a14 = a13;
  }
  const int8_t* a15 = (const int8_t*) ((uintptr_t) a14 + a_stride);
  if XNN_UNPREDICTABLE(mr != 16) {
    a15 = a14;
  }

  xnn_ace_load_palette2_config();

  do {
    const __m512i vksum0 = _mm512_loadu_epi32((const int32_t*) w + 0);
    const __m512i vksum1 = _mm512_loadu_epi32((const int32_t*) w + 16);
    w = (const int32_t*) w + 32;

    _tile_zero(0);
    _tile_zero(1);

    size_t k = 0;
    for (; k + 4 * sizeof(int8_t) <= kc; k += 4 * sizeof(int8_t)) {
      const __m512i va = _mm512_setr_epi32(
          unaligned_load_s32(a0 + k),
          unaligned_load_s32(a1 + k),
          unaligned_load_s32(a2 + k),
          unaligned_load_s32(a3 + k),
          unaligned_load_s32(a4 + k),
          unaligned_load_s32(a5 + k),
          unaligned_load_s32(a6 + k),
          unaligned_load_s32(a7 + k),
          unaligned_load_s32(a8 + k),
          unaligned_load_s32(a9 + k),
          unaligned_load_s32(a10 + k),
          unaligned_load_s32(a11 + k),
          unaligned_load_s32(a12 + k),
          unaligned_load_s32(a13 + k),
          unaligned_load_s32(a14 + k),
          unaligned_load_s32(a15 + k));
      const __m512i vb0 = _mm512_loadu_epi32((const int8_t*) w);
      xnn_ace_top4bssd_tmm0(va, vb0);
      if XNN_LIKELY(nc > 16) {
        const __m512i vb1 = _mm512_loadu_epi32((const int8_t*) w + 64);
        xnn_ace_top4bssd_tmm1(va, vb1);
      }
      w = (const int8_t*) w + 128;
    }
    if XNN_UNLIKELY(k < kc) {
      const __mmask16 vkmask = _cvtu32_mask16((UINT32_C(1) << (kc - k)) - UINT32_C(1));
      const __m512i va = _mm512_setr_epi32(
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a0 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a1 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a2 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a3 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a4 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a5 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a6 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a7 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a8 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a9 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a10 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a11 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a12 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a13 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a14 + k)),
          _mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a15 + k)));
      const __m512i vb0 = _mm512_loadu_epi32((const int8_t*) w);
      xnn_ace_top4bssd_tmm0(va, vb0);
      if XNN_LIKELY(nc > 16) {
        const __m512i vb1 = _mm512_loadu_epi32((const int8_t*) w + 64);
        xnn_ace_top4bssd_tmm1(va, vb1);
      }
      w = (const int8_t*) w + 128;
    }

    const __m512 vfilter_output_scale0 = _mm512_loadu_ps((const float*) w + 0);
    const __m512 vfilter_output_scale1 = _mm512_loadu_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;
    const __m512 vbias0 = _mm512_loadu_ps((const float*) w + 0);
    const __m512 vbias1 = _mm512_loadu_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;

    float* c0 = c;
    for (size_t m = 0; m < mr; ++m) {
      const __m512i vzero_point = _mm512_set1_epi32((int) quantization_params[m].zero_point);
      const __m512 vinv_scale = _mm512_set1_ps(quantization_params[m].inv_scale);
      if XNN_LIKELY(nc >= 32) {
        __m512i vacc0 = _mm512_mullo_epi32(vksum0, vzero_point);
        __m512i vacc1 = _mm512_mullo_epi32(vksum1, vzero_point);
        vacc0 = _mm512_add_epi32(vacc0, xnn_ace_tilemovrow_tmm0((uint32_t) m));
        vacc1 = _mm512_add_epi32(vacc1, xnn_ace_tilemovrow_tmm1((uint32_t) m));
        __m512 vscaled0 = _mm512_cvtepi32_ps(vacc0);
        __m512 vscaled1 = _mm512_cvtepi32_ps(vacc1);
        vscaled0 = _mm512_mul_ps(vscaled0, vinv_scale);
        vscaled1 = _mm512_mul_ps(vscaled1, vinv_scale);
        vscaled0 = _mm512_fmadd_ps(vscaled0, vfilter_output_scale0, vbias0);
        vscaled1 = _mm512_fmadd_ps(vscaled1, vfilter_output_scale1, vbias1);
        vscaled0 = _mm512_max_ps(vscaled0, voutput_min);
        vscaled1 = _mm512_max_ps(vscaled1, voutput_min);
        vscaled0 = _mm512_min_ps(vscaled0, voutput_max);
        vscaled1 = _mm512_min_ps(vscaled1, voutput_max);
        _mm512_storeu_ps(c0 + 0, vscaled0);
        _mm512_storeu_ps(c0 + 16, vscaled1);
      } else {
        __m512i vacc0 = _mm512_mullo_epi32(vksum0, vzero_point);
        vacc0 = _mm512_add_epi32(vacc0, xnn_ace_tilemovrow_tmm0((uint32_t) m));
        __m512 vscaled0 = _mm512_cvtepi32_ps(vacc0);
        vscaled0 = _mm512_mul_ps(vscaled0, vinv_scale);
        vscaled0 = _mm512_fmadd_ps(vscaled0, vfilter_output_scale0, vbias0);
        vscaled0 = _mm512_max_ps(vscaled0, voutput_min);
        vscaled0 = _mm512_min_ps(vscaled0, voutput_max);
        const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - UINT64_C(1)) >> 0) & 0xFFFF));
        _mm512_mask_storeu_ps(c0 + 0, vmask0, vscaled0);
        if (nc > 16) {
        __m512i vacc1 = _mm512_mullo_epi32(vksum1, vzero_point);
        vacc1 = _mm512_add_epi32(vacc1, xnn_ace_tilemovrow_tmm1((uint32_t) m));
        __m512 vscaled1 = _mm512_cvtepi32_ps(vacc1);
        vscaled1 = _mm512_mul_ps(vscaled1, vinv_scale);
        vscaled1 = _mm512_fmadd_ps(vscaled1, vfilter_output_scale1, vbias1);
        vscaled1 = _mm512_max_ps(vscaled1, voutput_min);
        vscaled1 = _mm512_min_ps(vscaled1, voutput_max);
        const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - UINT64_C(1)) >> 16) & 0xFFFF));
        _mm512_mask_storeu_ps(c0 + 16, vmask1, vscaled1);
        }
      }
      c0 = (float*) ((uintptr_t) c0 + cm_stride);
    }

    if XNN_LIKELY(nc >= 32) {
      c = (float*) ((uintptr_t) c + cn_stride);
      nc -= 32;
    } else {
      nc = 0;
    }
  } while (nc != 0);

  _tile_release();
#else
  (void) mr;
  (void) nc;
  (void) kc;
  (void) a;
  (void) a_stride;
  (void) w;
  (void) c;
  (void) cm_stride;
  (void) cn_stride;
  (void) params;
  (void) quantization_params;
#endif  // XNN_ARCH_X86_64
}
