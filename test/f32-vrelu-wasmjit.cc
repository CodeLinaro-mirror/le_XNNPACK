// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vrelu-wasmjit.yaml
//   Generator: tools/generate-vunary-test.py


#include <vector>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/vunary.h>

#include "vunary-microkernel-tester.h"


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X1_NO_LOCAL, batch_eq_1) {
    VUnaryMicrokernelTester()
      .batch_size(1)
      .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 1, 0);
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X1_NO_LOCAL, batch_gt_1) {
    for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 1, 0);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X1_NO_LOCAL, inplace) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 1, 0);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X2_NO_LOCAL, batch_eq_2) {
    VUnaryMicrokernelTester()
      .batch_size(2)
      .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 2, 0);
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X2_NO_LOCAL, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 2, 0);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X2_NO_LOCAL, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 2, 0);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X2_NO_LOCAL, batch_gt_2) {
    for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 2, 0);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X2_NO_LOCAL, inplace) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 2, 0);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X4_NO_LOCAL, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 4, 0);
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X4_NO_LOCAL, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 4, 0);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X4_NO_LOCAL, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 4, 0);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X4_NO_LOCAL, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 4, 0);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X4_NO_LOCAL, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 4, 0);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X8_NO_LOCAL, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 8, 0);
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X8_NO_LOCAL, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 8, 0);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X8_NO_LOCAL, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 8, 0);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X8_NO_LOCAL, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 8, 0);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X8_NO_LOCAL, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 8, 0);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X1_WITH_LOCAL, batch_eq_1) {
    VUnaryMicrokernelTester()
      .batch_size(1)
      .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 1, 1);
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X1_WITH_LOCAL, batch_gt_1) {
    for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 1, 1);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X1_WITH_LOCAL, inplace) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 1, 1);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X2_WITH_LOCAL, batch_eq_2) {
    VUnaryMicrokernelTester()
      .batch_size(2)
      .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 2, 1);
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X2_WITH_LOCAL, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 2, 1);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X2_WITH_LOCAL, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 2, 1);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X2_WITH_LOCAL, batch_gt_2) {
    for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 2, 1);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X2_WITH_LOCAL, inplace) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 2, 1);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X4_WITH_LOCAL, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 4, 1);
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X4_WITH_LOCAL, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 4, 1);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X4_WITH_LOCAL, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 4, 1);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X4_WITH_LOCAL, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 4, 1);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X4_WITH_LOCAL, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 4, 1);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X8_WITH_LOCAL, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 8, 1);
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X8_WITH_LOCAL, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 8, 1);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X8_WITH_LOCAL, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 8, 1);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X8_WITH_LOCAL, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 8, 1);
    }
  }

  TEST(GENERATE_F32_VRELU__JIT_WASM32_SHR_X8_WITH_LOCAL, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 8, 1);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
