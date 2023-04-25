// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/microparams-init.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, k_eq_1) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(1)
      .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, strided_cn) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, k_eq_1_strided_a) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(1)
      .a_stride(3)
      .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, k_eq_1_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, k_eq_1_subtile_m) {
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, k_eq_1_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, k_gt_1_strided_a) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, k_gt_1_subtile) {
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, n_gt_8_strided_a) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, n_div_8_strided_a) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(7)
          .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, strided_cm_subtile) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, qmin) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, qmax) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, strided_cm) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat, xnn_init_f32_minmax_wasmsimd_params);
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD
