// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xmmintrin.h>

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/transpose.h>

void xnn_x32_transpose_ukernel__4x4_sse(
    const uint32_t *input,
    uint32_t * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height){
  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_wbytes = tile_width * sizeof(uint32_t);
  assert(block_height >= tile_height);
  assert(block_width >= tile_width);
  assert(output_stride >= block_height * sizeof(uint32_t));
  assert(input_stride >= block_width * sizeof(uint32_t));
  size_t bw = block_width;
  const size_t input_reset = tile_wbytes - block_height * input_stride;
  const size_t output_col_reset = tile_wbytes - 3 * output_stride;
  const size_t output_row_reset = tile_height * output_stride - block_height * sizeof(uint32_t);
  const float *i = (const float*)input;
  float *o = (float*)output;
  size_t bh = block_height;
  for (; bw >= tile_width; bw -= tile_width) {
    bh = block_height;
inner_loop:
    for (; bh >= tile_height; bh -= tile_height) {
      __m128 v0 = _mm_loadu_ps(i);
      i = (float*) ((uintptr_t) i + input_stride);
      __m128 v1 = _mm_loadu_ps(i);
      i = (float*) ((uintptr_t) i + input_stride);
      __m128 v2 = _mm_loadu_ps(i);
      i = (float*) ((uintptr_t) i + input_stride);
      __m128 v3 = _mm_loadu_ps(i);
      i = (float*) ((uintptr_t) i + input_stride);
      _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
      _mm_storeu_ps(o, v0);
      o = (float*) ((uintptr_t) o + output_stride);
      _mm_storeu_ps(o, v1);
      o = (float*) ((uintptr_t) o + output_stride);
      _mm_storeu_ps(o, v2);
      o = (float*) ((uintptr_t) o + output_stride);
      _mm_storeu_ps(o, v3);
      o = (float*) ((uintptr_t) o + output_col_reset);
    }
    if XNN_UNLIKELY(bh != 0) {
      const size_t address_increment = tile_height - bh;
      i = (float*) ((uintptr_t) i - address_increment * input_stride);
      o = (float*) ((uintptr_t) o - address_increment * sizeof(uint32_t));
      bh = tile_height;
      goto inner_loop;
    }
    i = (float*) ((uintptr_t) i + input_reset);
    o = (float*) ((uintptr_t) o + output_row_reset);
  }
  if XNN_UNLIKELY(bw != 0) {
    // Shift input and output pointers back.
    const size_t address_increment = tile_width - bw;
    i = (float*) ((uintptr_t) i -  address_increment * sizeof(uint32_t));
    o = (float*) ((uintptr_t) o -  address_increment * (output_stride));
    bh = block_height;
    bw = tile_width;
    goto inner_loop;
  }
}
