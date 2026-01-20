// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/unary/reference.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

template <typename A, typename X>
void TestOp(A, X, const unary_op_info& op_info, ynn_unary_operator op) {
  ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> rank_dist(1, YNN_MAX_TENSOR_RANK);
  for (auto _ : FuzzTest(std::chrono::milliseconds(100))) {
    size_t rank = rank_dist(rng);
    constexpr size_t max_size = 1024;
    const size_t max_dim = static_cast<size_t>(std::ceil(
        std::pow(static_cast<double>(max_size),
                 1.0 / static_cast<double>(std::max<size_t>(1, rank)))));
    quantization_params a_quantization = random_quantization(A(), rng);
    quantization_params output_quantization = random_quantization(X(), rng);

    SubgraphBuilder subgraph(2);
    subgraph.AddInput(type_of<A>(), rank, 0, a_quantization)
        .AddOutput(type_of<X>(), rank, 1, output_quantization)
        .AddUnary(op, 0, 1);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank, 1, max_dim);

      Tensor<A> a(shape);
      Tensor<X> output(shape);

      const unary_op_info& op_info = *get_unary_op_info(op);
      interval domain = op_info.domain(type_of<A>());
      TypeGenerator<A> a_gen(domain.min, domain.max, a_quantization);
      a.generate([&]() { return a_gen(rng); });

      runtime.ReshapeExternalTensor(shape, a.data(), 0).ReshapeRuntime();

      ASSERT_EQ(runtime.GetExternalTensorShape(1), shape);

      runtime.SetupExternalTensor(output.data(), 1).InvokeRuntime();
      check_results(op_info, a, output, a_quantization, output_quantization);
    }
  }
}

template <typename T>
void TestOp(ynn_unary_operator op) {
  const unary_op_info& op_info = *get_unary_op_info(op);
  TestOp(quantized<T>(), quantized<T>(), op_info, op);
}

class UnaryFallbackTest : public testing::TestWithParam<ynn_unary_operator> {};

TEST_P(UnaryFallbackTest, Int8) { TestOp<int8_t>(GetParam()); }
TEST_P(UnaryFallbackTest, Uint8) { TestOp<uint8_t>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(
    UnaryFallbackTest, UnaryFallbackTest,
    ::testing::Values(ynn_unary_exp, ynn_unary_expm1, ynn_unary_erf,
                      ynn_unary_log, ynn_unary_log1p, ynn_unary_square,
                      ynn_unary_square_root, ynn_unary_reciprocal_square_root,
                      ynn_unary_tanh, ynn_unary_cube_root, ynn_unary_sine,
                      ynn_unary_cosine, ynn_unary_sigmoid, ynn_unary_hardswish),
    [](const testing::TestParamInfo<UnaryFallbackTest::ParamType>& info) {
      return to_string(info.param);
    });

}  // namespace ynn
