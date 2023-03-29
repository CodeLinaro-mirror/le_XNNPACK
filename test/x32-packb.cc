// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x32-packb.yaml
//   Generator: tools/generate-packb-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/packb.h>
#include "packb-microkernel-tester.h"


TEST(X32_PACKB_GEMM_X2__SCALAR_FLOAT, n_eq_2) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .Test(xnn_x32_packb_gemm_ukernel_x2__scalar_float);
  }
}

TEST(X32_PACKB_GEMM_X2__SCALAR_FLOAT, n_div_2) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .n(4)
      .k(k)
      .nr(2)
      .Test(xnn_x32_packb_gemm_ukernel_x2__scalar_float);
  }
}

TEST(X32_PACKB_GEMM_X2__SCALAR_FLOAT, n_lt_2) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 2; n++) {
      PackBMicrokernelTester()
        .n(n)
        .k(k)
        .nr(2)
        .Test(xnn_x32_packb_gemm_ukernel_x2__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_X2__SCALAR_FLOAT, n_gt_2) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 3; n < 4; n++) {
      PackBMicrokernelTester()
        .n(n)
        .k(k)
        .nr(2)
        .Test(xnn_x32_packb_gemm_ukernel_x2__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_X2__SCALAR_FLOAT, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      PackBMicrokernelTester()
        .g(g)
        .n(2)
        .k(k)
        .nr(2)
        .Test(xnn_x32_packb_gemm_ukernel_x2__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_X2__SCALAR_INT, n_eq_2) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .Test(xnn_x32_packb_gemm_ukernel_x2__scalar_int);
  }
}

TEST(X32_PACKB_GEMM_X2__SCALAR_INT, n_div_2) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .n(4)
      .k(k)
      .nr(2)
      .Test(xnn_x32_packb_gemm_ukernel_x2__scalar_int);
  }
}

TEST(X32_PACKB_GEMM_X2__SCALAR_INT, n_lt_2) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 2; n++) {
      PackBMicrokernelTester()
        .n(n)
        .k(k)
        .nr(2)
        .Test(xnn_x32_packb_gemm_ukernel_x2__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_X2__SCALAR_INT, n_gt_2) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 3; n < 4; n++) {
      PackBMicrokernelTester()
        .n(n)
        .k(k)
        .nr(2)
        .Test(xnn_x32_packb_gemm_ukernel_x2__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_X2__SCALAR_INT, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      PackBMicrokernelTester()
        .g(g)
        .n(2)
        .k(k)
        .nr(2)
        .Test(xnn_x32_packb_gemm_ukernel_x2__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_X4__SCALAR_FLOAT, n_eq_4) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .Test(xnn_x32_packb_gemm_ukernel_x4__scalar_float);
  }
}

TEST(X32_PACKB_GEMM_X4__SCALAR_FLOAT, n_div_4) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .n(8)
      .k(k)
      .nr(4)
      .Test(xnn_x32_packb_gemm_ukernel_x4__scalar_float);
  }
}

TEST(X32_PACKB_GEMM_X4__SCALAR_FLOAT, n_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 4; n++) {
      PackBMicrokernelTester()
        .n(n)
        .k(k)
        .nr(4)
        .Test(xnn_x32_packb_gemm_ukernel_x4__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_X4__SCALAR_FLOAT, n_gt_4) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 5; n < 8; n++) {
      PackBMicrokernelTester()
        .n(n)
        .k(k)
        .nr(4)
        .Test(xnn_x32_packb_gemm_ukernel_x4__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_X4__SCALAR_FLOAT, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      PackBMicrokernelTester()
        .g(g)
        .n(4)
        .k(k)
        .nr(4)
        .Test(xnn_x32_packb_gemm_ukernel_x4__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_X4__SCALAR_INT, n_eq_4) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .Test(xnn_x32_packb_gemm_ukernel_x4__scalar_int);
  }
}

TEST(X32_PACKB_GEMM_X4__SCALAR_INT, n_div_4) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .n(8)
      .k(k)
      .nr(4)
      .Test(xnn_x32_packb_gemm_ukernel_x4__scalar_int);
  }
}

TEST(X32_PACKB_GEMM_X4__SCALAR_INT, n_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 4; n++) {
      PackBMicrokernelTester()
        .n(n)
        .k(k)
        .nr(4)
        .Test(xnn_x32_packb_gemm_ukernel_x4__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_X4__SCALAR_INT, n_gt_4) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 5; n < 8; n++) {
      PackBMicrokernelTester()
        .n(n)
        .k(k)
        .nr(4)
        .Test(xnn_x32_packb_gemm_ukernel_x4__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_X4__SCALAR_INT, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      PackBMicrokernelTester()
        .g(g)
        .n(4)
        .k(k)
        .nr(4)
        .Test(xnn_x32_packb_gemm_ukernel_x4__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_X8__SCALAR_FLOAT, n_eq_8) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .Test(xnn_x32_packb_gemm_ukernel_x8__scalar_float);
  }
}

TEST(X32_PACKB_GEMM_X8__SCALAR_FLOAT, n_div_8) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .n(16)
      .k(k)
      .nr(8)
      .Test(xnn_x32_packb_gemm_ukernel_x8__scalar_float);
  }
}

TEST(X32_PACKB_GEMM_X8__SCALAR_FLOAT, n_lt_8) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 8; n++) {
      PackBMicrokernelTester()
        .n(n)
        .k(k)
        .nr(8)
        .Test(xnn_x32_packb_gemm_ukernel_x8__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_X8__SCALAR_FLOAT, n_gt_8) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 9; n < 16; n++) {
      PackBMicrokernelTester()
        .n(n)
        .k(k)
        .nr(8)
        .Test(xnn_x32_packb_gemm_ukernel_x8__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_X8__SCALAR_FLOAT, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      PackBMicrokernelTester()
        .g(g)
        .n(8)
        .k(k)
        .nr(8)
        .Test(xnn_x32_packb_gemm_ukernel_x8__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_X8__SCALAR_INT, n_eq_8) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .Test(xnn_x32_packb_gemm_ukernel_x8__scalar_int);
  }
}

TEST(X32_PACKB_GEMM_X8__SCALAR_INT, n_div_8) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .n(16)
      .k(k)
      .nr(8)
      .Test(xnn_x32_packb_gemm_ukernel_x8__scalar_int);
  }
}

TEST(X32_PACKB_GEMM_X8__SCALAR_INT, n_lt_8) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 8; n++) {
      PackBMicrokernelTester()
        .n(n)
        .k(k)
        .nr(8)
        .Test(xnn_x32_packb_gemm_ukernel_x8__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_X8__SCALAR_INT, n_gt_8) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 9; n < 16; n++) {
      PackBMicrokernelTester()
        .n(n)
        .k(k)
        .nr(8)
        .Test(xnn_x32_packb_gemm_ukernel_x8__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_X8__SCALAR_INT, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 4; k++) {
      PackBMicrokernelTester()
        .g(g)
        .n(8)
        .k(k)
        .nr(8)
        .Test(xnn_x32_packb_gemm_ukernel_x8__scalar_int);
    }
  }
}