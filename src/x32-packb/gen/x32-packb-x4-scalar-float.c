// Auto-generated file. Do not edit!
//   Template: src/x32-packb/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/math.h>
#include <xnnpack/packb.h>

void xnn_x32_packb_gemm_ukernel_x4__scalar_float(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  const uint32_t* bias,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const union xnn_x32_packb_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(nr == 4);   // This kernel is for NR=4
  assert(packed_weights != NULL);

  float* out = (float*) packed_weights;
  const float* b = (const float*) bias;
  do {
    // NC main loop multiple of 4
    size_t n = nc;
    for (;n >= 4; n -= 4) {
      out[0] = b[0];
      out[1] = b[1];
      out[2] = b[2];
      out[3] = b[3];
      b += 4;
      out += 4 + 4 * kc;
      out = (float*) ((uintptr_t) out + extra_bytes);
    }

    // NC remainder (1..3)
    if XNN_UNLIKELY(n != 0) {
      size_t nb = n;
      do {
        *out++ = *b++;
      } while (--nb != 0);
      out += (4 - n) + 4 * kc;
      out = (float*) ((uintptr_t) out + extra_bytes);
    }
  } while (--g != 0);
}
