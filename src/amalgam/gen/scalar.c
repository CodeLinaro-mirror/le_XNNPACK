// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <fxdiv.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <xnnpack/common.h>
#include <xnnpack/gemm.h>
#include <xnnpack/lut.h>
#include <xnnpack/math.h>
#include <xnnpack/packw.h>
#include <xnnpack/transpose.h>
#include <xnnpack/unaligned.h>
#include <xnnpack/vunary.h>


void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__scalar(
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
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;

  do {
    const int32_t vksum0 = unaligned_indexed_load_s32(w, 0);
    const int32_t vksum1 = unaligned_indexed_load_s32(w, 1);
    const int32_t vinput_zero_point0 = quantization_params[0].zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point0;
    int32_t vacc0x1 = vksum1 * vinput_zero_point0;
    w = (const int32_t*) w + 2;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      w = (const int8_t*) w + 2;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;

    const float vascale0 = quantization_params[0].inv_scale;
    const float vbscale0 = unaligned_indexed_load_f32(w, 0);
    const float vbscale1 = unaligned_indexed_load_f32(w, 1);
    const float vbias0 = unaligned_indexed_load_f32(w, 2);
    vout0x0 = math_muladd_f32(vout0x0, vascale0 * vbscale0, vbias0);
    const float vbias1 = unaligned_indexed_load_f32(w, 3);
    vout0x1 = math_muladd_f32(vout0x1, vascale0 * vbscale1, vbias1);

    w = (const float*) w + 4;

    const float voutput_min = params->scalar.min;
    vout0x0 = math_max_f32(vout0x0, voutput_min);
    vout0x1 = math_max_f32(vout0x1, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = math_min_f32(vout0x0, voutput_max);
    vout0x1 = math_min_f32(vout0x1, voutput_max);

    if XNN_LIKELY(nc >= 2) {
      c0[0] = vout0x0;
      c0[1] = vout0x1;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 2;
    } else {
      if (nc & 1) {
        c0[0] = vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

static inline uint32_t compute_sum(
    size_t n,
    const uint8_t* x,
    const uint32_t* t)
{
  assert(n != 0);

  uint32_t vsum = 0;
  do {
    const size_t vx = *x++;
    vsum += t[vx];
  } while (--n != 0);
  return vsum;
}

void xnn_u8_lut32norm_ukernel__scalar(
    size_t n,
    const uint8_t* x,
    const uint32_t* t,
    uint8_t* y)
{
  assert(n != 0);

  const uint32_t vsum = compute_sum(n, x, t);
  assert(vsum != 0);

  struct fxdiv_divisor_uint32_t vsum_divisor = fxdiv_init_uint32_t(vsum);
  const uint32_t vrounding = (vsum >> 1);
  do {
    const size_t vx = *x++;
    const uint32_t vt = t[vx];
    const uint32_t vq = fxdiv_quotient_uint32_t((vt << 8) + vrounding, vsum_divisor);
    const uint8_t vy = vq > 255 ? UINT8_C(255) : (uint8_t) vq;
    *y++ = vy;
  } while (--n != 0);
}

void xnn_x24_transposec_ukernel__1x2_scalar(
    const void *input,
    void * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height,
    const union xnn_x24_transpose_params* params) XNN_OOB_READS
{
  assert(output_stride >= block_height * 3);
  assert(input_stride >= block_width * 3);

  const size_t input_reset = 6 - block_height * input_stride;
  const size_t output_reset = 2 * output_stride - block_height * 3;
  const size_t input_offset = 1 * input_stride;

  const uint8_t* i0 = (const uint8_t*) input;

  uint8_t* o0 = (uint8_t*) output;
  uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + output_stride);

  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 1; bh -= 1) {
      o1[0] = i0[3];
      o1[1] = i0[4];
      o1[2] = i0[5];
      o1 += 3;
      o0[0] = i0[0];
      o0[1] = i0[1];
      o0[2] = i0[2];
      o0 += 3;
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }

    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    o0 = (uint8_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint8_t*) ((uintptr_t) o1 + output_reset);
    block_width = doz(block_width, 2);
  } while (block_width != 0);
}

void xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint32_t* weights,
  const uint32_t* bias,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 2);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  float* out = (float*) packed_weights;
  const float* b = (const float*) bias;

  do {
    // NC main loop multiple of 2
    const float* w0 = (const float*) weights;
    size_t n = nc;
    for (;n >= 2; n -= 2) {
      if XNN_LIKELY(b != NULL) {
        out[0] = b[0];
        out[1] = b[1];
        b += 2;
      } else {
        out[0] = 0;
        out[1] = 0;
      }
      out += 2;

      const float* w1 = w0 + kc;

      // KC main loop multiple of 2x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const float v00 = w0[0];
        const float v01 = w0[1];
        const float v02 = w0[2];
        const float v03 = w0[3];
        w0 += 4;
        const float v10 = w1[0];
        const float v11 = w1[1];
        const float v12 = w1[2];
        const float v13 = w1[3];
        w1 += 4;
        out[0] = v00;
        out[1] = v10;
        out[2] = v01;
        out[3] = v11;
        out[4] = v02;
        out[5] = v12;
        out[6] = v03;
        out[7] = v13;
        out += 8;
      }

      // KC remainder
      for (; k != 0; --k) {
        const float v0 = *w0++;
        out[0] = v0;
        const float v1 = *w1++;
        out[1] = v1;
        out += 2;
      }
      out = (float*) ((uintptr_t) out + extra_bytes);
      w0 = w1;
    }

    // NC remainder (1..1)
    if XNN_UNLIKELY(n != 0) {
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *out++ = *b++;
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
          *out++ = 0;
        } while (--nb != 0);
      }
      out += (2 - n);


      // KC main loop multiple of 2x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const float v00 = w0[0];
        const float v01 = w0[1];
        const float v02 = w0[2];
        const float v03 = w0[3];
        w0 += 4;
        out[0] = v00;
        out[2] = v01;
        out[4] = v02;
        out[6] = v03;
        out += 8;
      }

      // KC remainder of 1..3
      for (; k != 0; --k) {
        const float v0 = *w0++;
        out[0] = v0;
        out += 2;
      }
      out = (float*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}

void xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint32_t* weights,
  const uint32_t* bias,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 4);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  float* out = (float*) packed_weights;
  const float* b = (const float*) bias;

  do {
    // NC main loop multiple of 4
    const float* w0 = (const float*) weights;
    size_t n = nc;
    for (;n >= 4; n -= 4) {
      if XNN_LIKELY(b != NULL) {
        out[0] = b[0];
        out[1] = b[1];
        out[2] = b[2];
        out[3] = b[3];
        b += 4;
      } else {
        out[0] = 0;
        out[1] = 0;
        out[2] = 0;
        out[3] = 0;
      }
      out += 4;

      const float* w1 = w0 + kc;
      const float* w2 = w1 + kc;
      const float* w3 = w2 + kc;

      // KC main loop multiple of 4x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const float v00 = w0[0];
        const float v01 = w0[1];
        const float v02 = w0[2];
        const float v03 = w0[3];
        w0 += 4;
        const float v10 = w1[0];
        const float v11 = w1[1];
        const float v12 = w1[2];
        const float v13 = w1[3];
        w1 += 4;
        const float v20 = w2[0];
        const float v21 = w2[1];
        const float v22 = w2[2];
        const float v23 = w2[3];
        w2 += 4;
        const float v30 = w3[0];
        const float v31 = w3[1];
        const float v32 = w3[2];
        const float v33 = w3[3];
        w3 += 4;
        out[0] = v00;
        out[1] = v10;
        out[2] = v20;
        out[3] = v30;
        out[4] = v01;
        out[5] = v11;
        out[6] = v21;
        out[7] = v31;
        out[8] = v02;
        out[9] = v12;
        out[10] = v22;
        out[11] = v32;
        out[12] = v03;
        out[13] = v13;
        out[14] = v23;
        out[15] = v33;
        out += 16;
      }

      // KC remainder
      for (; k != 0; --k) {
        const float v0 = *w0++;
        out[0] = v0;
        const float v1 = *w1++;
        out[1] = v1;
        const float v2 = *w2++;
        out[2] = v2;
        const float v3 = *w3++;
        out[3] = v3;
        out += 4;
      }
      out = (float*) ((uintptr_t) out + extra_bytes);
      w0 = w3;
    }

    // NC remainder (1..3)
    if XNN_UNLIKELY(n != 0) {
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *out++ = *b++;
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
          *out++ = 0;
        } while (--nb != 0);
      }
      out += (4 - n);

      // NR remainder has less than 4 rows so last row is not loaded
      const float* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const float* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }

      // KC main loop multiple of 4x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const float v00 = w0[0];
        const float v01 = w0[1];
        const float v02 = w0[2];
        const float v03 = w0[3];
        w0 += 4;
        const float v10 = w1[0];
        const float v11 = w1[1];
        const float v12 = w1[2];
        const float v13 = w1[3];
        w1 += 4;
        const float v20 = w2[0];
        const float v21 = w2[1];
        const float v22 = w2[2];
        const float v23 = w2[3];
        w2 += 4;
        out[0] = v00;
        out[1] = v10;
        out[2] = v20;
        out[4] = v01;
        out[5] = v11;
        out[6] = v21;
        out[8] = v02;
        out[9] = v12;
        out[10] = v22;
        out[12] = v03;
        out[13] = v13;
        out[14] = v23;
        out += 16;
      }

      // KC remainder of 1..3
      for (; k != 0; --k) {
        const float v0 = *w0++;
        out[0] = v0;
        const float v1 = *w1++;
        out[1] = v1;
        const float v2 = *w2++;
        out[2] = v2;
        out += 4;
      }
      out = (float*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}

void xnn_x8_lut_ukernel__scalar_x4(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t table[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    const size_t vx0 = (size_t) input[0];
    const size_t vx1 = (size_t) input[1];
    const size_t vx2 = (size_t) input[2];
    const size_t vx3 = (size_t) input[3];
    input += 4;

    const uint32_t vt0 = (uint32_t) table[vx0];
    const uint32_t vt1 = (uint32_t) table[vx1];
    const uint32_t vt2 = (uint32_t) table[vx2];
    const uint32_t vt3 = (uint32_t) table[vx3];

    output[0] = (uint8_t) vt0;
    output[1] = (uint8_t) vt1;
    output[2] = (uint8_t) vt2;
    output[3] = (uint8_t) vt3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const size_t vx = (size_t) *input++;
      const uint32_t vt = (uint32_t) table[vx];
      *output++ = (uint8_t) vt;
      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}

void xnn_xx_copy_ukernel__scalar_memcpy(size_t batch, const void* input, void* output, const void* params) {
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);

  memcpy(output, input, batch);
}

void xnn_xx_transposev_ukernel__1x1_scalar_memcpy(
    const void* input,
    void* output,
    size_t input_row_stride,
    size_t output_row_stride,
    size_t input_element_stride,
    size_t output_element_stride,
    size_t element_size,
    size_t block_width,
    size_t block_height)
{
  const size_t input_reset = input_element_stride - block_height * input_row_stride;
  const size_t output_reset = output_row_stride - block_height * output_element_stride;

  const void* i = (const void*) input;
  void* o = (void*) output;

  do {
    size_t bh = block_height;
    for (; bh >= 1; bh -= 1) {
      memcpy(o, i, element_size);
      i = (const void*) ((uintptr_t) i + input_row_stride);
      o = (void*) ((uintptr_t) o + output_element_stride);
    }

    i = (const void*) ((uintptr_t) i + input_reset);
    o = (void*) ((uintptr_t) o + output_reset);
    block_width -= 1;
  } while (block_width != 0);
}
