// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

template <typename AT, typename BT, typename CT>
void Reference(Tensor<AT> a, Tensor<BT> b, Tensor<CT> c) {
  using B_info = type_info<BT>;

  // This helper allows omitting 2 of the 3 k dimensions. Canonicalize to 3 k
  // dimensions here.
  while (a.rank() < 4 && b.rank() < 4) {
    a = a.expand_dims({1});
    b = b.expand_dims({0});
  }

  const size_t M = c.extent(0);
  const size_t N = c.extent(1);
  ASSERT_EQ(c.rank(), 2);
  ASSERT_EQ(M, a.extent(0));
  ASSERT_EQ(N, b.extent(3));
  const size_t K3 = a.extent(1);
  const size_t K2 = a.extent(2);
  const size_t K1 = a.extent(3);
  ASSERT_EQ(b.extent(0), K3);
  ASSERT_EQ(b.extent(1), K2);
  ASSERT_EQ(b.extent(2), K1);

  for (size_t i = 0; i < M; ++i) {
    CT* c_i = &c(i, 0);
    for (size_t k3 = 0; k3 < K3; ++k3) {
      for (size_t k2 = 0; k2 < K2; ++k2) {
        for (size_t k1 = 0; k1 < K1; ++k1) {
          const CT a_ik = static_cast<CT>(a(i, k3, k2, k1));
          const BT* b_k1 = address_of(b(k3, k2, k1, 0));
          for (size_t j = 0; j < N; ++j) {
            c_i[j] = c_i[j] + a_ik * static_cast<CT>(B_info::get(b_k1, j));
          }
        }
      }
    }
  }
}

TEST(DotTest, FusedDotTilingMismatch) {
  ReplicableRandomDevice rng;

  const float max_abs_value = 1.0f;
  const size_t Batch = 1024;
  const size_t M = 1;
  const size_t N = 64;
  const size_t K = 32;

  std::vector<size_t> a_shape = {Batch, M, K};
  std::vector<size_t> b_shape = {K, N};
  std::vector<size_t> d_shape = {Batch, M, N};
  std::vector<size_t> output_shape = {Batch, M, N};

  Tensor<float> a(a_shape);
  Tensor<float> b(b_shape);
  Tensor<float> d(d_shape);
  Tensor<float> expected(output_shape);

  fill_random(a.data(), a.size(), rng, -max_abs_value, max_abs_value);
  fill_random(b.data(), b.size(), rng, -max_abs_value, max_abs_value);
  fill_random(d.data(), d.size(), rng, -max_abs_value, max_abs_value);

  // expected = d + dot(a, b)
  expected.assign(d);
  for (size_t b_idx = 0; b_idx < Batch; ++b_idx) {
    Tensor<float> a_slice = a.slice(0, b_idx).remove_dim(0);
    Tensor<float> expected_slice = expected.slice(0, b_idx).remove_dim(0);
    Reference(a_slice, b, expected_slice);
  }

  // 4 external values: A (0), B (1), D (2), Output (3)
  SubgraphBuilder subgraph(4, 0);

  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t d_id = 2;
  const uint32_t output_id = 3;

  subgraph.AddInput(ynn_type_fp32, a_shape, a_id)
      .AddTensor(b, b_id)
      .AddInput(ynn_type_fp32, d_shape, d_id)
      .AddOutput(ynn_type_fp32, output_shape, output_id);

  // Define internal tensor for dot output
  uint32_t dot_out_id = YNN_INVALID_VALUE_ID;
  subgraph.AddTensor(ynn_type_fp32, output_shape, dot_out_id);
  ASSERT_NE(dot_out_id, YNN_INVALID_VALUE_ID);

  // Add nodes:
  // dot_out = dot(a, b)
  // output = add(dot_out, d)
  subgraph.AddDot(1, a_id, b_id, YNN_INVALID_VALUE_ID, dot_out_id)
      .AddBinary(ynn_binary_add, dot_out_id, d_id, output_id);

  // Run the subgraph
  TestScheduler scheduler(3);
  Runtime runtime(subgraph.GetSubgraph(), &scheduler);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor(a_shape, a.data(), a_id);
  runtime.ReshapeExternalTensor(d_shape, d.data(), d_id);

  runtime.ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  ASSERT_EQ(runtime.GetExternalTensorShape(output_id), output_shape);

  Tensor<float> c(output_shape);

  runtime.SetupExternalTensor(c.data(), output_id).InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  // Compare results
  for (const auto& i : EnumerateIndices(output_shape)) {
    ASSERT_NEAR(c(i), expected(i), 1e-4) << "i=" << index_to_string(i);
  }
}

}  // namespace ynn
