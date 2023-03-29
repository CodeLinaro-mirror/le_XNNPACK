// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/pack.h>

class PackBMicrokernelTester {
 public:

  inline PackBMicrokernelTester& g(size_t g) {
    this->g_ = g;
    return *this;
  }

  inline size_t g() const {
    return this->g_;
  }

  inline PackBMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }

  inline PackBMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline size_t packed_n() const {
    return round_up(n(), nr());
  }

  inline PackBMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  inline size_t k() const {
    return this->k_;
  }

  void Test(xnn_x32_packb_gemm_ukernel_fn packb) const {
    std::vector<uint32_t> weights(g() * n() * k());
    std::vector<uint32_t> bias(g() * n());
    std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> packed_w(g() * (packed_n() * k() + packed_n()));
    std::vector<uint32_t> packed_w_ref(g() * (packed_n() * k() + packed_n()));

    std::fill(weights.begin(), weights.end(), 0xDEADBEEF);
    std::iota(bias.begin(), bias.end(), UINT32_C(0x80000000));
    std::fill(packed_w.begin(), packed_w.end(), UINT32_C(0x12345678));
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), UINT32_C(0xDEADBEEF));

    // Compute reference results.
    xnn_pack_f32_gemm_goi_w(
        g(), n(), k(), nr(), /*kr=*/1, /*sr=*/1, reinterpret_cast<const float*>(weights.data()),
        reinterpret_cast<const float*>(bias.data()), reinterpret_cast<float*>(packed_w_ref.data()), 0, nullptr);

    // Call optimized micro-kernel.
    packb(g(), n(), k(), nr(), bias.data(), packed_w.data(), 0, nullptr);

    // Verify results.
    for (size_t i = 0; i < g() * (packed_n() * k() + packed_n()); i++) {
      if (packed_w_ref[i] !=  UINT32_C(0xDEADBEEF)) {  // Allow weights and padding to differ.
        EXPECT_EQ(packed_w[i], packed_w_ref[i])
            << "at n " << i << " of " << (g() * (packed_n() * k() + packed_n()));
      }
    }
  }

  void Test(xnn_x32_zerob_gemm_ukernel_fn zerob) const {
    std::vector<uint32_t> weights(g() * n() * k());
    std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> packed_w(g() * (packed_n() * k() + packed_n()));
    std::vector<uint32_t> packed_w_ref(g() * (packed_n() * k() + packed_n()));

    std::fill(weights.begin(), weights.end(), 0xDEADBEEF);
    std::fill(packed_w.begin(), packed_w.end(), UINT32_C(0x12345678));
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), UINT32_C(0xDEADBEEF));

    // Compute reference results.
    xnn_pack_f32_gemm_goi_w(
        g(), n(), k(), nr(), /*kr=*/1, /*sr=*/1, reinterpret_cast<const float*>(weights.data()),
        nullptr, reinterpret_cast<float*>(packed_w_ref.data()), 0, nullptr);

    // Call optimized micro-kernel.
    zerob(g(), n(), k(), nr(), packed_w.data(), 0, nullptr);

    // Verify results.
    for (size_t i = 0; i < g() * (packed_n() * k() + packed_n()); i++) {
      if (packed_w_ref[i] !=  UINT32_C(0xDEADBEEF)) {  // Allow weights and padding to differ.
        EXPECT_EQ(packed_w[i], packed_w_ref[i])
            << "at n " << i << " of " << (g() * (packed_n() * k() + packed_n()));
        EXPECT_EQ(packed_w_ref[i], 0.0f)
            << "at n " << i << " of " << (g() * (packed_n() * k() + packed_n()));
      }
    }
  }

 private:
  size_t g_{1};
  size_t n_{1};
  size_t nr_{1};
  size_t k_{1};
};
