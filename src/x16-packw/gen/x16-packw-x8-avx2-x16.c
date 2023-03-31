// Auto-generated file. Do not edit!
//   Template: src/x16-packw/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <string.h>

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include <xnnpack/packw.h>










void xnn_x16_packw_gemm_goi_ukernel_x8__avx2_x16(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* weights,
  const uint16_t* bias,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{

  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 8);   // This kernel is for NR=8
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);


  uint16_t buf[16];

  do {
    const uint16_t* w0 = weights;
    size_t n = nc;
    for (; n >= 8; n -= 8) {
      __m128i vtmp;
      if XNN_LIKELY(bias != NULL) {
        vtmp = _mm_loadu_si128((const __m128i*) bias);
        bias += 8;
      } else {
        vtmp = _mm_set1_epi32(0);
      }
      _mm_storeu_si128((__m128i*) packed_weights, vtmp);
      packed_weights += 8;

      const uint16_t* row0 = w0 + 0 * kc;
      const uint16_t* row1 = w0 + 1 * kc;
      const uint16_t* row2 = w0 + 2 * kc;
      const uint16_t* row3 = w0 + 3 * kc;
      const uint16_t* row4 = w0 + 4 * kc;
      const uint16_t* row5 = w0 + 5 * kc;
      const uint16_t* row6 = w0 + 6 * kc;
      const uint16_t* row7 = w0 + 7 * kc;
      size_t k = kc;
      for (; k >= 16; k -= 16) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*)row0);
        row0 += 16;
        __m256i v1 = _mm256_loadu_si256((const __m256i*)row1);
        row1 += 16;
        __m256i v2 = _mm256_loadu_si256((const __m256i*)row2);
        row2 += 16;
        __m256i v3 = _mm256_loadu_si256((const __m256i*)row3);
        row3 += 16;
        __m256i v4 = _mm256_loadu_si256((const __m256i*)row4);
        row4 += 16;
        __m256i v5 = _mm256_loadu_si256((const __m256i*)row5);
        row5 += 16;
        __m256i v6 = _mm256_loadu_si256((const __m256i*)row6);
        row6 += 16;
        __m256i v7 = _mm256_loadu_si256((const __m256i*)row7);
        row7 += 16;
        // Interleave 16-bit lanes
        __m256i vt0 = _mm256_unpacklo_epi16(v0, v1);
        __m256i vt1 = _mm256_unpackhi_epi16(v0, v1);
        __m256i vt2 = _mm256_unpacklo_epi16(v2, v3);
        __m256i vt3 = _mm256_unpackhi_epi16(v2, v3);
        __m256i vt4 = _mm256_unpacklo_epi16(v4, v5);
        __m256i vt5 = _mm256_unpackhi_epi16(v4, v5);
        __m256i vt6 = _mm256_unpacklo_epi16(v6, v7);
        __m256i vt7 = _mm256_unpackhi_epi16(v6, v7);

        // Interleave 32-bit lanes
        v0 = _mm256_unpacklo_epi32(vt0, vt2);
        v1 = _mm256_unpackhi_epi32(vt0, vt2);
        v2 = _mm256_unpacklo_epi32(vt1, vt3);
        v3 = _mm256_unpackhi_epi32(vt1, vt3);
        v4 = _mm256_unpacklo_epi32(vt4, vt6);
        v5 = _mm256_unpackhi_epi32(vt4, vt6);
        v6 = _mm256_unpacklo_epi32(vt5, vt7);
        v7 = _mm256_unpackhi_epi32(vt5, vt7);

        // Interleave 64-bit lanes
        vt0 = _mm256_unpacklo_epi64(v0, v4);
        vt1 = _mm256_unpackhi_epi64(v0, v4);
        vt2 = _mm256_unpacklo_epi64(v1, v5);
        vt3 = _mm256_unpackhi_epi64(v1, v5);
        vt4 = _mm256_unpacklo_epi64(v2, v6);
        vt5 = _mm256_unpackhi_epi64(v2, v6);
        vt6 = _mm256_unpacklo_epi64(v3, v7);
        vt7 = _mm256_unpackhi_epi64(v3, v7);

        v0 = _mm256_permute2f128_si256(vt0, vt1, 0x20);
        v1 = _mm256_permute2f128_si256(vt0, vt1, 0x31);
        v2 = _mm256_permute2f128_si256(vt2, vt3, 0x20);
        v3 = _mm256_permute2f128_si256(vt2, vt3, 0x31);
        v4 = _mm256_permute2f128_si256(vt4, vt5, 0x20);
        v5 = _mm256_permute2f128_si256(vt4, vt5, 0x31);
        v6 = _mm256_permute2f128_si256(vt6, vt7, 0x20);
        v7 = _mm256_permute2f128_si256(vt6, vt7, 0x31);
        _mm256_storeu_si256((__m256i*) packed_weights, v0);
        packed_weights += 16;
        _mm256_storeu_si256((__m256i*) packed_weights, v2);
        packed_weights += 16;
        _mm256_storeu_si256((__m256i*) packed_weights, v4);
        packed_weights += 16;
        _mm256_storeu_si256((__m256i*) packed_weights, v6);
        packed_weights += 16;
        _mm256_storeu_si256((__m256i*) packed_weights, v1);
        packed_weights += 16;
        _mm256_storeu_si256((__m256i*) packed_weights, v3);
        packed_weights += 16;
        _mm256_storeu_si256((__m256i*) packed_weights, v5);
        packed_weights += 16;
        _mm256_storeu_si256((__m256i*) packed_weights, v7);
        packed_weights += 16;
      }
      // KC remainder
      if XNN_UNLIKELY(k != 0) {
        memcpy(buf, row0, k * 2);
        __m256i v0 = _mm256_loadu_si256((const __m256i*)buf);
        row0 += k;
        row0 += k;
        memcpy(buf, row1, k * 2);
        __m256i v1 = _mm256_loadu_si256((const __m256i*)buf);
        row1 += k;
        row1 += k;
        memcpy(buf, row2, k * 2);
        __m256i v2 = _mm256_loadu_si256((const __m256i*)buf);
        row2 += k;
        row2 += k;
        memcpy(buf, row3, k * 2);
        __m256i v3 = _mm256_loadu_si256((const __m256i*)buf);
        row3 += k;
        row3 += k;
        memcpy(buf, row4, k * 2);
        __m256i v4 = _mm256_loadu_si256((const __m256i*)buf);
        row4 += k;
        row4 += k;
        memcpy(buf, row5, k * 2);
        __m256i v5 = _mm256_loadu_si256((const __m256i*)buf);
        row5 += k;
        row5 += k;
        memcpy(buf, row6, k * 2);
        __m256i v6 = _mm256_loadu_si256((const __m256i*)buf);
        row6 += k;
        row6 += k;
        memcpy(buf, row7, k * 2);
        __m256i v7 = _mm256_loadu_si256((const __m256i*)buf);
        row7 += k;
        // Interleave 16-bit lanes
        __m256i vt0 = _mm256_unpacklo_epi16(v0, v1);
        __m256i vt1 = _mm256_unpackhi_epi16(v0, v1);
        __m256i vt2 = _mm256_unpacklo_epi16(v2, v3);
        __m256i vt3 = _mm256_unpackhi_epi16(v2, v3);
        __m256i vt4 = _mm256_unpacklo_epi16(v4, v5);
        __m256i vt5 = _mm256_unpackhi_epi16(v4, v5);
        __m256i vt6 = _mm256_unpacklo_epi16(v6, v7);
        __m256i vt7 = _mm256_unpackhi_epi16(v6, v7);

        // Interleave 32-bit lanes
        v0 = _mm256_unpacklo_epi32(vt0, vt2);
        v1 = _mm256_unpackhi_epi32(vt0, vt2);
        v2 = _mm256_unpacklo_epi32(vt1, vt3);
        v3 = _mm256_unpackhi_epi32(vt1, vt3);
        v4 = _mm256_unpacklo_epi32(vt4, vt6);
        v5 = _mm256_unpackhi_epi32(vt4, vt6);
        v6 = _mm256_unpacklo_epi32(vt5, vt7);
        v7 = _mm256_unpackhi_epi32(vt5, vt7);

        // Interleave 64-bit lanes
        vt0 = _mm256_unpacklo_epi64(v0, v4);
        vt1 = _mm256_unpackhi_epi64(v0, v4);
        vt2 = _mm256_unpacklo_epi64(v1, v5);
        vt3 = _mm256_unpackhi_epi64(v1, v5);
        vt4 = _mm256_unpacklo_epi64(v2, v6);
        vt5 = _mm256_unpackhi_epi64(v2, v6);
        vt6 = _mm256_unpacklo_epi64(v3, v7);
        vt7 = _mm256_unpackhi_epi64(v3, v7);

        v0 = _mm256_permute2f128_si256(vt0, vt1, 0x20);
        v1 = _mm256_permute2f128_si256(vt0, vt1, 0x31);
        v2 = _mm256_permute2f128_si256(vt2, vt3, 0x20);
        v3 = _mm256_permute2f128_si256(vt2, vt3, 0x31);
        v4 = _mm256_permute2f128_si256(vt4, vt5, 0x20);
        v5 = _mm256_permute2f128_si256(vt4, vt5, 0x31);
        v6 = _mm256_permute2f128_si256(vt6, vt7, 0x20);
        v7 = _mm256_permute2f128_si256(vt6, vt7, 0x31);
        if (1 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v0, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (2 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v0);
          packed_weights += 16;
        }
        if (3 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v2, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (4 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v2);
          packed_weights += 16;
        }
        if (5 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v4, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (6 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v4);
          packed_weights += 16;
        }
        if (7 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v6, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (8 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v6);
          packed_weights += 16;
        }
        if (9 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v1, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (10 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v1);
          packed_weights += 16;
        }
        if (11 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v3, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (12 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v3);
          packed_weights += 16;
        }
        if (13 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v5, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (14 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v5);
          packed_weights += 16;
        }
        if (15 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v7, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (16 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v7);
          packed_weights += 16;
        }
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
      w0 = row7;
    }


    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 7);
      if XNN_LIKELY(bias != NULL) {
        memcpy(packed_weights, bias, n * 2);
        bias += n;
      } else {
        memset(packed_weights, 0, 16);
      }
      packed_weights += 8;
      // NR remainder has less than 8 rows so last row is not loaded
      const uint16_t* row0 = w0;
      const uint16_t* row1 = row0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        row1 = row0;
      }
      const uint16_t* row2 = row1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        row2 = row1;
      }
      const uint16_t* row3 = row2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        row3 = row2;
      }
      const uint16_t* row4 = row3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        row4 = row3;
      }
      const uint16_t* row5 = row4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        row5 = row4;
      }
      const uint16_t* row6 = row5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        row6 = row5;
      }

      size_t k = kc;
      for (; k >= 16; k -= 16) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*)row0);
        row0 += 16;
        __m256i v1 = _mm256_loadu_si256((const __m256i*)row1);
        row1 += 16;
        __m256i v2 = _mm256_loadu_si256((const __m256i*)row2);
        row2 += 16;
        __m256i v3 = _mm256_loadu_si256((const __m256i*)row3);
        row3 += 16;
        __m256i v4 = _mm256_loadu_si256((const __m256i*)row4);
        row4 += 16;
        __m256i v5 = _mm256_loadu_si256((const __m256i*)row5);
        row5 += 16;
        __m256i v6 = _mm256_loadu_si256((const __m256i*)row6);
        row6 += 16;
        __m256i v7;
        // Interleave 16-bit lanes
        __m256i vt0 = _mm256_unpacklo_epi16(v0, v1);
        __m256i vt1 = _mm256_unpackhi_epi16(v0, v1);
        __m256i vt2 = _mm256_unpacklo_epi16(v2, v3);
        __m256i vt3 = _mm256_unpackhi_epi16(v2, v3);
        __m256i vt4 = _mm256_unpacklo_epi16(v4, v5);
        __m256i vt5 = _mm256_unpackhi_epi16(v4, v5);
        __m256i vt6 = _mm256_unpacklo_epi16(v6, v6);
        __m256i vt7 = _mm256_unpackhi_epi16(v6, v6);

        // Interleave 32-bit lanes
        v0 = _mm256_unpacklo_epi32(vt0, vt2);
        v1 = _mm256_unpackhi_epi32(vt0, vt2);
        v2 = _mm256_unpacklo_epi32(vt1, vt3);
        v3 = _mm256_unpackhi_epi32(vt1, vt3);
        v4 = _mm256_unpacklo_epi32(vt4, vt6);
        v5 = _mm256_unpackhi_epi32(vt4, vt6);
        v6 = _mm256_unpacklo_epi32(vt5, vt7);
        v7 = _mm256_unpackhi_epi32(vt5, vt7);

        // Interleave 64-bit lanes
        vt0 = _mm256_unpacklo_epi64(v0, v4);
        vt1 = _mm256_unpackhi_epi64(v0, v4);
        vt2 = _mm256_unpacklo_epi64(v1, v5);
        vt3 = _mm256_unpackhi_epi64(v1, v5);
        vt4 = _mm256_unpacklo_epi64(v2, v6);
        vt5 = _mm256_unpackhi_epi64(v2, v6);
        vt6 = _mm256_unpacklo_epi64(v3, v7);
        vt7 = _mm256_unpackhi_epi64(v3, v7);

        v0 = _mm256_permute2f128_si256(vt0, vt1, 0x20);
        v1 = _mm256_permute2f128_si256(vt0, vt1, 0x31);
        v2 = _mm256_permute2f128_si256(vt2, vt3, 0x20);
        v3 = _mm256_permute2f128_si256(vt2, vt3, 0x31);
        v4 = _mm256_permute2f128_si256(vt4, vt5, 0x20);
        v5 = _mm256_permute2f128_si256(vt4, vt5, 0x31);
        v6 = _mm256_permute2f128_si256(vt6, vt7, 0x20);
        v7 = _mm256_permute2f128_si256(vt6, vt7, 0x31);
        _mm256_storeu_si256((__m256i*) packed_weights, v0);
        packed_weights += 16;
        _mm256_storeu_si256((__m256i*) packed_weights, v2);
        packed_weights += 16;
        _mm256_storeu_si256((__m256i*) packed_weights, v4);
        packed_weights += 16;
        _mm256_storeu_si256((__m256i*) packed_weights, v6);
        packed_weights += 16;
        _mm256_storeu_si256((__m256i*) packed_weights, v1);
        packed_weights += 16;
        _mm256_storeu_si256((__m256i*) packed_weights, v3);
        packed_weights += 16;
        _mm256_storeu_si256((__m256i*) packed_weights, v5);
        packed_weights += 16;
        _mm256_storeu_si256((__m256i*) packed_weights, v7);
        packed_weights += 16;
      }

      // KC and NC remainder
      if XNN_UNLIKELY(k != 0) {
        memcpy(buf, row0, k * 2);
        __m256i v0 = _mm256_loadu_si256((const __m256i*)buf);
        row0 += k;
        memcpy(buf, row1, k * 2);
        __m256i v1 = _mm256_loadu_si256((const __m256i*)buf);
        row1 += k;
        memcpy(buf, row2, k * 2);
        __m256i v2 = _mm256_loadu_si256((const __m256i*)buf);
        row2 += k;
        memcpy(buf, row3, k * 2);
        __m256i v3 = _mm256_loadu_si256((const __m256i*)buf);
        row3 += k;
        memcpy(buf, row4, k * 2);
        __m256i v4 = _mm256_loadu_si256((const __m256i*)buf);
        row4 += k;
        memcpy(buf, row5, k * 2);
        __m256i v5 = _mm256_loadu_si256((const __m256i*)buf);
        row5 += k;
        memcpy(buf, row6, k * 2);
        __m256i v6 = _mm256_loadu_si256((const __m256i*)buf);
        row6 += k;
        __m256i v7;
        // Interleave 16-bit lanes
        __m256i vt0 = _mm256_unpacklo_epi16(v0, v1);
        __m256i vt1 = _mm256_unpackhi_epi16(v0, v1);
        __m256i vt2 = _mm256_unpacklo_epi16(v2, v3);
        __m256i vt3 = _mm256_unpackhi_epi16(v2, v3);
        __m256i vt4 = _mm256_unpacklo_epi16(v4, v5);
        __m256i vt5 = _mm256_unpackhi_epi16(v4, v5);
        __m256i vt6 = _mm256_unpacklo_epi16(v6, v6);
        __m256i vt7 = _mm256_unpackhi_epi16(v6, v6);

        // Interleave 32-bit lanes
        v0 = _mm256_unpacklo_epi32(vt0, vt2);
        v1 = _mm256_unpackhi_epi32(vt0, vt2);
        v2 = _mm256_unpacklo_epi32(vt1, vt3);
        v3 = _mm256_unpackhi_epi32(vt1, vt3);
        v4 = _mm256_unpacklo_epi32(vt4, vt6);
        v5 = _mm256_unpackhi_epi32(vt4, vt6);
        v6 = _mm256_unpacklo_epi32(vt5, vt7);
        v7 = _mm256_unpackhi_epi32(vt5, vt7);

        // Interleave 64-bit lanes
        vt0 = _mm256_unpacklo_epi64(v0, v4);
        vt1 = _mm256_unpackhi_epi64(v0, v4);
        vt2 = _mm256_unpacklo_epi64(v1, v5);
        vt3 = _mm256_unpackhi_epi64(v1, v5);
        vt4 = _mm256_unpacklo_epi64(v2, v6);
        vt5 = _mm256_unpackhi_epi64(v2, v6);
        vt6 = _mm256_unpacklo_epi64(v3, v7);
        vt7 = _mm256_unpackhi_epi64(v3, v7);

        v0 = _mm256_permute2f128_si256(vt0, vt1, 0x20);
        v1 = _mm256_permute2f128_si256(vt0, vt1, 0x31);
        v2 = _mm256_permute2f128_si256(vt2, vt3, 0x20);
        v3 = _mm256_permute2f128_si256(vt2, vt3, 0x31);
        v4 = _mm256_permute2f128_si256(vt4, vt5, 0x20);
        v5 = _mm256_permute2f128_si256(vt4, vt5, 0x31);
        v6 = _mm256_permute2f128_si256(vt6, vt7, 0x20);
        v7 = _mm256_permute2f128_si256(vt6, vt7, 0x31);
        if (1 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v0, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (2 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v0);
          packed_weights += 16;
        }
        if (3 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v2, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (4 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v2);
          packed_weights += 16;
        }
        if (5 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v4, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (6 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v4);
          packed_weights += 16;
        }
        if (7 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v6, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (8 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v6);
          packed_weights += 16;
        }
        if (9 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v1, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (10 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v1);
          packed_weights += 16;
        }
        if (11 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v3, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (12 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v3);
          packed_weights += 16;
        }
        if (13 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v5, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (14 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v5);
          packed_weights += 16;
        }
        if (15 == k) {
          __m128i vtlow = _mm256_extracti128_si256(v7, 0);
          _mm_storeu_si128((__m128i*) packed_weights, vtlow);
          packed_weights += 8;
        }
        if (16 <= k) {
          _mm256_storeu_si256((__m256i*) packed_weights, v7);
          packed_weights += 16;
        }
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
