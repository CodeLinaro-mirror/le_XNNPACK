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

void xnn_x32_packb_gemm_ukernel_x2__scalar_int(
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
  assert(nr == 2);   // This kernel is for NR=2
  assert(packed_weights != NULL);

  uint32_t* out = (uint32_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;
  do {
    // NC main loop multiple of 2
    size_t n = nc;
    for (;n >= 2; n -= 2) {
      out[0] = b[0];
      out[1] = b[1];
      b += 2;
      out += 2 + 2 * kc;
      out = (uint32_t*) ((uintptr_t) out + extra_bytes);
    }

    // NC remainder (1..1)
    if XNN_UNLIKELY(n != 0) {
      size_t nb = n;
      do {
        *out++ = *b++;
      } while (--nb != 0);
      out += (2 - n) + 2 * kc;
      out = (uint32_t*) ((uintptr_t) out + extra_bytes);
    }
  } while (--g != 0);
}
