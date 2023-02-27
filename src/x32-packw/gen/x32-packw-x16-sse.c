// Auto-generated file. Do not edit!
//   Template: src/x32-packw/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <emmintrin.h>

#include <xnnpack/packw.h>


void xnn_x32_packw_gemm_goi_ukernel_x16__sse2(
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
  assert(nr == 16);   // This kernel is for NR=16
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 16
    const float* w0 = (const float*) weights;
    size_t n = nc;

    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(b != NULL) {
        const __m128 vb0 = _mm_loadu_ps(b + 0);
        const __m128 vb4 = _mm_loadu_ps(b + 4);
        const __m128 vb8 = _mm_loadu_ps(b + 8);
        const __m128 vb12 = _mm_loadu_ps(b + 12);
        _mm_store_ps(packed_w + 0, vb0);
        _mm_store_ps(packed_w + 4, vb4);
        _mm_store_ps(packed_w + 8, vb8);
        _mm_store_ps(packed_w + 12, vb12);
        b += 16;
      }
      packed_w += 16;

      const float* w1 = w0 + kc;
      const float* w2 = w1 + kc;
      const float* w3 = w2 + kc;
      const float* w4 = w3 + kc;
      const float* w5 = w4 + kc;
      const float* w6 = w5 + kc;
      const float* w7 = w6 + kc;
      const float* w8 = w7 + kc;
      const float* w9 = w8 + kc;
      const float* w10 = w9 + kc;
      const float* w11 = w10 + kc;
      const float* w12 = w11 + kc;
      const float* w13 = w12 + kc;
      const float* w14 = w13 + kc;
      const float* w15 = w14 + kc;

      // KC main loop multiple of 16x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 4x4
        // a b c d
        // e f g h
        // i j k l
        // m n o p
        __m128 v00 = _mm_loadu_ps(w0);
        w0 += 4;
        __m128 v10 = _mm_loadu_ps(w1);
        w1 += 4;
        __m128 v20 = _mm_loadu_ps(w2);
        w2 += 4;
        __m128 v30 = _mm_loadu_ps(w3);
        w3 += 4;
        __m128 v40 = _mm_loadu_ps(w4);
        w4 += 4;
        __m128 v50 = _mm_loadu_ps(w5);
        w5 += 4;
        __m128 v60 = _mm_loadu_ps(w6);
        w6 += 4;
        __m128 v70 = _mm_loadu_ps(w7);
        w7 += 4;
        __m128 v80 = _mm_loadu_ps(w8);
        w8 += 4;
        __m128 v90 = _mm_loadu_ps(w9);
        w9 += 4;
        __m128 v100 = _mm_loadu_ps(w10);
        w10 += 4;
        __m128 v110 = _mm_loadu_ps(w11);
        w11 += 4;
        __m128 v120 = _mm_loadu_ps(w12);
        w12 += 4;
        __m128 v130 = _mm_loadu_ps(w13);
        w13 += 4;
        __m128 v140 = _mm_loadu_ps(w14);
        w14 += 4;
        __m128 v150 = _mm_loadu_ps(w15);
        w15 += 4;

        // Transpose 2x2
        const __m128 t00 = _mm_unpacklo_ps(v00, v10);  // a e b f   from row 0, 1
        const __m128 t10 = _mm_unpacklo_ps(v20, v30);  // i m j n   from row 2, 3
        const __m128 t20 = _mm_unpackhi_ps(v00, v10);  // c g d h   from row 0, 1
        const __m128 t30 = _mm_unpackhi_ps(v20, v30);  // k o l p   from row 2, 3
        // Transpose 4x4
        v00 = (const __m128) _mm_unpacklo_pd((const __m128d) t00, (const __m128d) t10);  // a e i m   from row 0, 1
        v10 = (const __m128) _mm_unpackhi_pd((const __m128d) t00, (const __m128d) t10);  // b f j n   from row 0, 1
        v20 = (const __m128) _mm_unpacklo_pd((const __m128d) t20, (const __m128d) t30);  // c g k o   from row 2, 3
        v30 = (const __m128) _mm_unpackhi_pd((const __m128d) t20, (const __m128d) t30);  // d h l p   from row 2, 3
        // Transpose 2x2
        const __m128 t40 = _mm_unpacklo_ps(v40, v50);  // a e b f   from row 0, 1
        const __m128 t50 = _mm_unpacklo_ps(v60, v70);  // i m j n   from row 2, 3
        const __m128 t60 = _mm_unpackhi_ps(v40, v50);  // c g d h   from row 0, 1
        const __m128 t70 = _mm_unpackhi_ps(v60, v70);  // k o l p   from row 2, 3
        // Transpose 4x4
        v40 = (const __m128) _mm_unpacklo_pd((const __m128d) t40, (const __m128d) t50);  // a e i m   from row 0, 1
        v50 = (const __m128) _mm_unpackhi_pd((const __m128d) t40, (const __m128d) t50);  // b f j n   from row 0, 1
        v60 = (const __m128) _mm_unpacklo_pd((const __m128d) t60, (const __m128d) t70);  // c g k o   from row 2, 3
        v70 = (const __m128) _mm_unpackhi_pd((const __m128d) t60, (const __m128d) t70);  // d h l p   from row 2, 3
        // Transpose 2x2
        const __m128 t80 = _mm_unpacklo_ps(v80, v90);  // a e b f   from row 0, 1
        const __m128 t90 = _mm_unpacklo_ps(v100, v110);  // i m j n   from row 2, 3
        const __m128 t100 = _mm_unpackhi_ps(v80, v90);  // c g d h   from row 0, 1
        const __m128 t110 = _mm_unpackhi_ps(v100, v110);  // k o l p   from row 2, 3
        // Transpose 4x4
        v80 = (const __m128) _mm_unpacklo_pd((const __m128d) t80, (const __m128d) t90);  // a e i m   from row 0, 1
        v90 = (const __m128) _mm_unpackhi_pd((const __m128d) t80, (const __m128d) t90);  // b f j n   from row 0, 1
        v100 = (const __m128) _mm_unpacklo_pd((const __m128d) t100, (const __m128d) t110);  // c g k o   from row 2, 3
        v110 = (const __m128) _mm_unpackhi_pd((const __m128d) t100, (const __m128d) t110);  // d h l p   from row 2, 3
        // Transpose 2x2
        const __m128 t120 = _mm_unpacklo_ps(v120, v130);  // a e b f   from row 0, 1
        const __m128 t130 = _mm_unpacklo_ps(v140, v150);  // i m j n   from row 2, 3
        const __m128 t140 = _mm_unpackhi_ps(v120, v130);  // c g d h   from row 0, 1
        const __m128 t150 = _mm_unpackhi_ps(v140, v150);  // k o l p   from row 2, 3
        // Transpose 4x4
        v120 = (const __m128) _mm_unpacklo_pd((const __m128d) t120, (const __m128d) t130);  // a e i m   from row 0, 1
        v130 = (const __m128) _mm_unpackhi_pd((const __m128d) t120, (const __m128d) t130);  // b f j n   from row 0, 1
        v140 = (const __m128) _mm_unpacklo_pd((const __m128d) t140, (const __m128d) t150);  // c g k o   from row 2, 3
        v150 = (const __m128) _mm_unpackhi_pd((const __m128d) t140, (const __m128d) t150);  // d h l p   from row 2, 3

        _mm_store_ps(packed_w + 0, v00);
        _mm_store_ps(packed_w + 4, v40);
        _mm_store_ps(packed_w + 8, v80);
        _mm_store_ps(packed_w + 12, v120);
        _mm_store_ps(packed_w + 16, v10);
        _mm_store_ps(packed_w + 20, v50);
        _mm_store_ps(packed_w + 24, v90);
        _mm_store_ps(packed_w + 28, v130);
        _mm_store_ps(packed_w + 32, v20);
        _mm_store_ps(packed_w + 36, v60);
        _mm_store_ps(packed_w + 40, v100);
        _mm_store_ps(packed_w + 44, v140);
        _mm_store_ps(packed_w + 48, v30);
        _mm_store_ps(packed_w + 52, v70);
        _mm_store_ps(packed_w + 56, v110);
        _mm_store_ps(packed_w + 60, v150);
        packed_w += 64;
      }

      // KC remainder
      if XNN_UNLIKELY(k != 0) {
        for (; k >= 2; k -= 2) {
          // Read blocks of 4x2
          // a b
          // c d
          // e f
          // g h
          __m128 v0 = _mm_loadu_si64(w0);  w0 += 2;
          __m128 v1 = _mm_loadu_si64(w1);  w1 += 2;
          __m128 v2 = _mm_loadu_si64(w2);  w2 += 2;
          __m128 v3 = _mm_loadu_si64(w3);  w3 += 2;
          __m128 v4 = _mm_loadu_si64(w4);  w4 += 2;
          __m128 v5 = _mm_loadu_si64(w5);  w5 += 2;
          __m128 v6 = _mm_loadu_si64(w6);  w6 += 2;
          __m128 v7 = _mm_loadu_si64(w7);  w7 += 2;
          __m128 v8 = _mm_loadu_si64(w8);  w8 += 2;
          __m128 v9 = _mm_loadu_si64(w9);  w9 += 2;
          __m128 v10 = _mm_loadu_si64(w10);  w10 += 2;
          __m128 v11 = _mm_loadu_si64(w11);  w11 += 2;
          __m128 v12 = _mm_loadu_si64(w12);  w12 += 2;
          __m128 v13 = _mm_loadu_si64(w13);  w13 += 2;
          __m128 v14 = _mm_loadu_si64(w14);  w14 += 2;
          __m128 v15 = _mm_loadu_si64(w15);  w15 += 2;

          // Transpose 2x2
          const __m128 t0 = _mm_unpacklo_ps(v0, v1);  // a c b d   from row 0, 1
          const __m128 t1 = _mm_unpacklo_ps(v2, v3);  // e g f h   from row 2, 3
          // Transpose 4x4
          v0 = (const __m128) _mm_unpacklo_pd((const __m128d) t0, (const __m128d) t1);  // a c e g   from row 0, 1
          v1 = (const __m128) _mm_unpackhi_pd((const __m128d) t0, (const __m128d) t1);  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 t4 = _mm_unpacklo_ps(v4, v5);  // a c b d   from row 0, 1
          const __m128 t5 = _mm_unpacklo_ps(v6, v7);  // e g f h   from row 2, 3
          // Transpose 4x4
          v4 = (const __m128) _mm_unpacklo_pd((const __m128d) t4, (const __m128d) t5);  // a c e g   from row 0, 1
          v5 = (const __m128) _mm_unpackhi_pd((const __m128d) t4, (const __m128d) t5);  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 t8 = _mm_unpacklo_ps(v8, v9);  // a c b d   from row 0, 1
          const __m128 t9 = _mm_unpacklo_ps(v10, v11);  // e g f h   from row 2, 3
          // Transpose 4x4
          v8 = (const __m128) _mm_unpacklo_pd((const __m128d) t8, (const __m128d) t9);  // a c e g   from row 0, 1
          v9 = (const __m128) _mm_unpackhi_pd((const __m128d) t8, (const __m128d) t9);  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 t12 = _mm_unpacklo_ps(v12, v13);  // a c b d   from row 0, 1
          const __m128 t13 = _mm_unpacklo_ps(v14, v15);  // e g f h   from row 2, 3
          // Transpose 4x4
          v12 = (const __m128) _mm_unpacklo_pd((const __m128d) t12, (const __m128d) t13);  // a c e g   from row 0, 1
          v13 = (const __m128) _mm_unpackhi_pd((const __m128d) t12, (const __m128d) t13);  // b d f h   from row 0, 1

          _mm_store_ps(packed_w + 0, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          _mm_store_ps(packed_w + 16, v1);
          _mm_store_ps(packed_w + 20, v5);
          _mm_store_ps(packed_w + 24, v9);
          _mm_store_ps(packed_w + 28, v13);
          packed_w += 32;
        }
        if (k == 1) {
          // Read blocks of 4x1
          // a
          // b
          // c
          // d
          __m128 v0 = _mm_load_ss(w0);  w0 += 1;
          __m128 v1 = _mm_load_ss(w1);  w1 += 1;
          __m128 v2 = _mm_load_ss(w2);  w2 += 1;
          __m128 v3 = _mm_load_ss(w3);  w3 += 1;
          __m128 v4 = _mm_load_ss(w4);  w4 += 1;
          __m128 v5 = _mm_load_ss(w5);  w5 += 1;
          __m128 v6 = _mm_load_ss(w6);  w6 += 1;
          __m128 v7 = _mm_load_ss(w7);  w7 += 1;
          __m128 v8 = _mm_load_ss(w8);  w8 += 1;
          __m128 v9 = _mm_load_ss(w9);  w9 += 1;
          __m128 v10 = _mm_load_ss(w10);  w10 += 1;
          __m128 v11 = _mm_load_ss(w11);  w11 += 1;
          __m128 v12 = _mm_load_ss(w12);  w12 += 1;
          __m128 v13 = _mm_load_ss(w13);  w13 += 1;
          __m128 v14 = _mm_load_ss(w14);  w14 += 1;
          __m128 v15 = _mm_load_ss(w15);  w15 += 1;

          // Transpose 2x2
          const __m128 t0 = _mm_unpacklo_ps(v0, v1);  // a b  from row 0, 1
          const __m128 t1 = _mm_unpacklo_ps(v2, v3);  // c d  from row 2, 3
          // Transpose 4x4
          v0 = (const __m128) _mm_unpacklo_pd((const __m128d) t0, (const __m128d) t1);  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 t4 = _mm_unpacklo_ps(v4, v5);  // a b  from row 0, 1
          const __m128 t5 = _mm_unpacklo_ps(v6, v7);  // c d  from row 2, 3
          // Transpose 4x4
          v4 = (const __m128) _mm_unpacklo_pd((const __m128d) t4, (const __m128d) t5);  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 t8 = _mm_unpacklo_ps(v8, v9);  // a b  from row 0, 1
          const __m128 t9 = _mm_unpacklo_ps(v10, v11);  // c d  from row 2, 3
          // Transpose 4x4
          v8 = (const __m128) _mm_unpacklo_pd((const __m128d) t8, (const __m128d) t9);  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 t12 = _mm_unpacklo_ps(v12, v13);  // a b  from row 0, 1
          const __m128 t13 = _mm_unpacklo_ps(v14, v15);  // c d  from row 2, 3
          // Transpose 4x4
          v12 = (const __m128) _mm_unpacklo_pd((const __m128d) t12, (const __m128d) t13);  // a b c d   from row 0, 1

          _mm_store_ps(packed_w + 0, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          packed_w += 16;
        }
      }
      packed_w = (uint32_t*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w15;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++  = *b++;
        } while (--nb != 0);
        packed_w += (16 - n);
      } else {
        packed_w += 16;
      }

      // NR remainder has less than 16 rows so last row is not loaded
      // For SR=4 the
      const float* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const float* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const float* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const float* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const float* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const float* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const float* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const float* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const float* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const float* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const float* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const float* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const float* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const float* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }

      // KC main loop multiple of 16x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 4x4
        // a b c d
        // e f g h
        // i j k l
        // m n o p
        __m128 v00 = _mm_loadu_ps(w0);
        w0 += 4;
        __m128 v10 = _mm_loadu_ps(w1);
        w1 += 4;
        __m128 v20 = _mm_loadu_ps(w2);
        w2 += 4;
        __m128 v30 = _mm_loadu_ps(w3);
        w3 += 4;
        __m128 v40 = _mm_loadu_ps(w4);
        w4 += 4;
        __m128 v50 = _mm_loadu_ps(w5);
        w5 += 4;
        __m128 v60 = _mm_loadu_ps(w6);
        w6 += 4;
        __m128 v70 = _mm_loadu_ps(w7);
        w7 += 4;
        __m128 v80 = _mm_loadu_ps(w8);
        w8 += 4;
        __m128 v90 = _mm_loadu_ps(w9);
        w9 += 4;
        __m128 v100 = _mm_loadu_ps(w10);
        w10 += 4;
        __m128 v110 = _mm_loadu_ps(w11);
        w11 += 4;
        __m128 v120 = _mm_loadu_ps(w12);
        w12 += 4;
        __m128 v130 = _mm_loadu_ps(w13);
        w13 += 4;
        __m128 v140 = _mm_loadu_ps(w14);
        w14 += 4;

        __m128 v150 = _mm_setzero_ps();

        // Transpose 2x2
        const __m128 t00 = _mm_unpacklo_ps(v00, v10);  // a e b f   from row 0, 1
        const __m128 t10 = _mm_unpacklo_ps(v20, v30);  // i m j n   from row 2, 3
        const __m128 t20 = _mm_unpackhi_ps(v00, v10);  // c g d h   from row 0, 1
        const __m128 t30 = _mm_unpackhi_ps(v20, v30);  // k o l p   from row 2, 3
        // Transpose 4x4
        v00 = (const __m128) _mm_unpacklo_pd((const __m128d) t00, (const __m128d) t10);  // a e i m   from row 0, 1
        v10 = (const __m128) _mm_unpackhi_pd((const __m128d) t00, (const __m128d) t10);  // b f j n   from row 0, 1
        v20 = (const __m128) _mm_unpacklo_pd((const __m128d) t20, (const __m128d) t30);  // c g k o   from row 2, 3
        v30 = (const __m128) _mm_unpackhi_pd((const __m128d) t20, (const __m128d) t30);  // d h l p   from row 2, 3
        // Transpose 2x2
        const __m128 t40 = _mm_unpacklo_ps(v40, v50);  // a e b f   from row 0, 1
        const __m128 t50 = _mm_unpacklo_ps(v60, v70);  // i m j n   from row 2, 3
        const __m128 t60 = _mm_unpackhi_ps(v40, v50);  // c g d h   from row 0, 1
        const __m128 t70 = _mm_unpackhi_ps(v60, v70);  // k o l p   from row 2, 3
        // Transpose 4x4
        v40 = (const __m128) _mm_unpacklo_pd((const __m128d) t40, (const __m128d) t50);  // a e i m   from row 0, 1
        v50 = (const __m128) _mm_unpackhi_pd((const __m128d) t40, (const __m128d) t50);  // b f j n   from row 0, 1
        v60 = (const __m128) _mm_unpacklo_pd((const __m128d) t60, (const __m128d) t70);  // c g k o   from row 2, 3
        v70 = (const __m128) _mm_unpackhi_pd((const __m128d) t60, (const __m128d) t70);  // d h l p   from row 2, 3
        // Transpose 2x2
        const __m128 t80 = _mm_unpacklo_ps(v80, v90);  // a e b f   from row 0, 1
        const __m128 t90 = _mm_unpacklo_ps(v100, v110);  // i m j n   from row 2, 3
        const __m128 t100 = _mm_unpackhi_ps(v80, v90);  // c g d h   from row 0, 1
        const __m128 t110 = _mm_unpackhi_ps(v100, v110);  // k o l p   from row 2, 3
        // Transpose 4x4
        v80 = (const __m128) _mm_unpacklo_pd((const __m128d) t80, (const __m128d) t90);  // a e i m   from row 0, 1
        v90 = (const __m128) _mm_unpackhi_pd((const __m128d) t80, (const __m128d) t90);  // b f j n   from row 0, 1
        v100 = (const __m128) _mm_unpacklo_pd((const __m128d) t100, (const __m128d) t110);  // c g k o   from row 2, 3
        v110 = (const __m128) _mm_unpackhi_pd((const __m128d) t100, (const __m128d) t110);  // d h l p   from row 2, 3
        // Transpose 2x2
        const __m128 t120 = _mm_unpacklo_ps(v120, v130);  // a e b f   from row 0, 1
        const __m128 t130 = _mm_unpacklo_ps(v140, v150);  // i m j n   from row 2, 3
        const __m128 t140 = _mm_unpackhi_ps(v120, v130);  // c g d h   from row 0, 1
        const __m128 t150 = _mm_unpackhi_ps(v140, v150);  // k o l p   from row 2, 3
        // Transpose 4x4
        v120 = (const __m128) _mm_unpacklo_pd((const __m128d) t120, (const __m128d) t130);  // a e i m   from row 0, 1
        v130 = (const __m128) _mm_unpackhi_pd((const __m128d) t120, (const __m128d) t130);  // b f j n   from row 0, 1
        v140 = (const __m128) _mm_unpacklo_pd((const __m128d) t140, (const __m128d) t150);  // c g k o   from row 2, 3
        v150 = (const __m128) _mm_unpackhi_pd((const __m128d) t140, (const __m128d) t150);  // d h l p   from row 2, 3

        _mm_store_ps(packed_w + 0, v00);
        _mm_store_ps(packed_w + 4, v40);
        _mm_store_ps(packed_w + 8, v80);
        _mm_store_ps(packed_w + 12, v120);
        _mm_store_ps(packed_w + 16, v10);
        _mm_store_ps(packed_w + 20, v50);
        _mm_store_ps(packed_w + 24, v90);
        _mm_store_ps(packed_w + 28, v130);
        _mm_store_ps(packed_w + 32, v20);
        _mm_store_ps(packed_w + 36, v60);
        _mm_store_ps(packed_w + 40, v100);
        _mm_store_ps(packed_w + 44, v140);
        _mm_store_ps(packed_w + 48, v30);
        _mm_store_ps(packed_w + 52, v70);
        _mm_store_ps(packed_w + 56, v110);
        _mm_store_ps(packed_w + 60, v150);
        packed_w += 64;
      }

      // KC remainder
      if XNN_UNLIKELY(k != 0) {
        for (; k >= 2; k -= 2) {
          // Read blocks of 4x2
          // a b
          // c d
          // e f
          // g h
          __m128 v0 = _mm_loadu_si64(w0);  w0 += 2;
          __m128 v1 = _mm_loadu_si64(w1);  w1 += 2;
          __m128 v2 = _mm_loadu_si64(w2);  w2 += 2;
          __m128 v3 = _mm_loadu_si64(w3);  w3 += 2;
          __m128 v4 = _mm_loadu_si64(w4);  w4 += 2;
          __m128 v5 = _mm_loadu_si64(w5);  w5 += 2;
          __m128 v6 = _mm_loadu_si64(w6);  w6 += 2;
          __m128 v7 = _mm_loadu_si64(w7);  w7 += 2;
          __m128 v8 = _mm_loadu_si64(w8);  w8 += 2;
          __m128 v9 = _mm_loadu_si64(w9);  w9 += 2;
          __m128 v10 = _mm_loadu_si64(w10);  w10 += 2;
          __m128 v11 = _mm_loadu_si64(w11);  w11 += 2;
          __m128 v12 = _mm_loadu_si64(w12);  w12 += 2;
          __m128 v13 = _mm_loadu_si64(w13);  w13 += 2;
          __m128 v14 = _mm_loadu_si64(w14);  w14 += 2;
          __m128 v15 = _mm_setzero_ps();

          // Transpose 2x2
          const __m128 t0 = _mm_unpacklo_ps(v0, v1);  // a c b d   from row 0, 1
          const __m128 t1 = _mm_unpacklo_ps(v2, v3);  // e g f h   from row 2, 3
          // Transpose 4x4
          v0 = (const __m128) _mm_unpacklo_pd((const __m128d) t0, (const __m128d) t1);  // a c e g   from row 0, 1
          v1 = (const __m128) _mm_unpackhi_pd((const __m128d) t0, (const __m128d) t1);  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 t4 = _mm_unpacklo_ps(v4, v5);  // a c b d   from row 0, 1
          const __m128 t5 = _mm_unpacklo_ps(v6, v7);  // e g f h   from row 2, 3
          // Transpose 4x4
          v4 = (const __m128) _mm_unpacklo_pd((const __m128d) t4, (const __m128d) t5);  // a c e g   from row 0, 1
          v5 = (const __m128) _mm_unpackhi_pd((const __m128d) t4, (const __m128d) t5);  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 t8 = _mm_unpacklo_ps(v8, v9);  // a c b d   from row 0, 1
          const __m128 t9 = _mm_unpacklo_ps(v10, v11);  // e g f h   from row 2, 3
          // Transpose 4x4
          v8 = (const __m128) _mm_unpacklo_pd((const __m128d) t8, (const __m128d) t9);  // a c e g   from row 0, 1
          v9 = (const __m128) _mm_unpackhi_pd((const __m128d) t8, (const __m128d) t9);  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 t12 = _mm_unpacklo_ps(v12, v13);  // a c b d   from row 0, 1
          const __m128 t13 = _mm_unpacklo_ps(v14, v15);  // e g f h   from row 2, 3
          // Transpose 4x4
          v12 = (const __m128) _mm_unpacklo_pd((const __m128d) t12, (const __m128d) t13);  // a c e g   from row 0, 1
          v13 = (const __m128) _mm_unpackhi_pd((const __m128d) t12, (const __m128d) t13);  // b d f h   from row 0, 1

          _mm_store_ps(packed_w + 0, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          _mm_store_ps(packed_w + 16, v1);
          _mm_store_ps(packed_w + 20, v5);
          _mm_store_ps(packed_w + 24, v9);
          _mm_store_ps(packed_w + 28, v13);
          packed_w += 32;
        }
        if (k == 1) {
          // Read blocks of 4x1
          // a
          // b
          // c
          // d
          __m128 v0 = _mm_load_ss(w0);  w0 += 1;
          __m128 v1 = _mm_load_ss(w1);  w1 += 1;
          __m128 v2 = _mm_load_ss(w2);  w2 += 1;
          __m128 v3 = _mm_load_ss(w3);  w3 += 1;
          __m128 v4 = _mm_load_ss(w4);  w4 += 1;
          __m128 v5 = _mm_load_ss(w5);  w5 += 1;
          __m128 v6 = _mm_load_ss(w6);  w6 += 1;
          __m128 v7 = _mm_load_ss(w7);  w7 += 1;
          __m128 v8 = _mm_load_ss(w8);  w8 += 1;
          __m128 v9 = _mm_load_ss(w9);  w9 += 1;
          __m128 v10 = _mm_load_ss(w10);  w10 += 1;
          __m128 v11 = _mm_load_ss(w11);  w11 += 1;
          __m128 v12 = _mm_load_ss(w12);  w12 += 1;
          __m128 v13 = _mm_load_ss(w13);  w13 += 1;
          __m128 v14 = _mm_load_ss(w14);  w14 += 1;
          __m128 v15 = _mm_setzero_ps();

          // Transpose 2x2
          const __m128 t0 = _mm_unpacklo_ps(v0, v1);  // a b  from row 0, 1
          const __m128 t1 = _mm_unpacklo_ps(v2, v3);  // c d  from row 2, 3
          // Transpose 4x4
          v0 = (const __m128) _mm_unpacklo_pd((const __m128d) t0, (const __m128d) t1);  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 t4 = _mm_unpacklo_ps(v4, v5);  // a b  from row 0, 1
          const __m128 t5 = _mm_unpacklo_ps(v6, v7);  // c d  from row 2, 3
          // Transpose 4x4
          v4 = (const __m128) _mm_unpacklo_pd((const __m128d) t4, (const __m128d) t5);  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 t8 = _mm_unpacklo_ps(v8, v9);  // a b  from row 0, 1
          const __m128 t9 = _mm_unpacklo_ps(v10, v11);  // c d  from row 2, 3
          // Transpose 4x4
          v8 = (const __m128) _mm_unpacklo_pd((const __m128d) t8, (const __m128d) t9);  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 t12 = _mm_unpacklo_ps(v12, v13);  // a b  from row 0, 1
          const __m128 t13 = _mm_unpacklo_ps(v14, v15);  // c d  from row 2, 3
          // Transpose 4x4
          v12 = (const __m128) _mm_unpacklo_pd((const __m128d) t12, (const __m128d) t13);  // a b c d   from row 0, 1

          _mm_store_ps(packed_w + 0, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          packed_w += 16;
        }
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
