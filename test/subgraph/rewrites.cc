// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/subgraph.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/runtime-flags.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

namespace {

enum kTensorIDs {
  input_id = 0,
  output_id,
  extra_external_value_id,
  num_external_values,
};

template <typename T>
static Tensor<T> add_input_tensor(ReplicableRandomDevice& rng,
                                  SubgraphTester& subgraph,
                                  const TensorShape shape, uint32_t id) {
  Tensor<T> tensor(shape.dims, xnnpack::XnnExtraBytes);
  DatatypeGenerator<T> generator;
  tensor.generate([&]() { return generator(rng); });
  subgraph.AddInputTensor(shape, tensor.base(), id);
  return tensor;
}

template <typename T>
static std::tuple<Tensor<T>, uint32_t> add_static_tensor(
    ReplicableRandomDevice& rng, SubgraphTester& subgraph,
    const TensorShape shape, double min = 0.0, double max = 1.0) {
  Tensor<T> static_tensor(shape.dims, xnnpack::XnnExtraBytes);
  if (min < max) {
    DatatypeGenerator<T> generator(min, max);
    static_tensor.generate([&]() { return generator(rng); });
  } else {
    std::fill(static_tensor.begin(), static_tensor.end(), static_cast<T>(min));
  }
  uint32_t static_value_id = XNN_INVALID_VALUE_ID;
  subgraph.AddInternalStaticTensor(shape, xnn_datatype_of<T>(),
                                   &static_value_id, static_tensor.base(),
                                   /*flags=*/0);
  return {static_tensor, static_value_id};
}

template <typename T>
static uint32_t add_internal_dynamic_tensor(SubgraphTester& subgraph,
                                            TensorShape shape) {
  uint32_t value_id = XNN_INVALID_VALUE_ID;
  subgraph.AddInternalDynamicTensor(shape, xnn_datatype_of<T>(), &value_id,
                                    /*flags=*/0);
  return value_id;
}

static std::vector<size_t> create_random_extra_dims(ReplicableRandomDevice& rng,
                                                    size_t rank) {
  std::vector<size_t> indices(rank + 1);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);
  indices.resize(std::uniform_int_distribution<int>(
      1, std::min<int>(XNN_MAX_TENSOR_DIMS - rank, indices.size()))(rng));
  std::sort(indices.begin(), indices.end());
  return indices;
}

template <class T, class U>
std::pair<T, T> random_swap(ReplicableRandomDevice& rng, T a, U b) {
  return (rng() % 2) ? std::pair<T, T>{a, b} : std::pair<T, T>{b, a};
}
}  // namespace

template <typename F>
void RewriteTestImpl(
    size_t rank, F populate, int expected_size_diff,
    const std::map<enum xnn_node_type, int> expected_node_type_counts = {}) {
  ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> dim_dist(1, 9);

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(100))) {
    std::vector<size_t> input_shape = random_shape(rng, rank);

    // Create the subgraph inputs.
    DatatypeGenerator<float> generator(-1, 1);
    Tensor<float> input(input_shape, xnnpack::XnnExtraBytes);
    input.generate([&]() { return generator(rng); });

    // Define the subgraph.
    SubgraphTester subgraph(num_external_values);
    subgraph.AddInputTensor(input_shape, xnn_datatype_of<float>(), input_id);
    subgraph.AddOutputTensor(rank, xnn_datatype_of<float>(), output_id);
    populate(rng, subgraph);

    // Create a runtime without any rewrites.
    ASSERT_EQ(subgraph.CreateRuntime(
                  /*threadpool=*/nullptr,
                  xnn_test_runtime_flags() | XNN_FLAG_NO_OPERATOR_FUSION),
              xnn_status_success);
    const size_t num_nodes_orig = subgraph.NumNodes();

    // Attach the inputs.
    subgraph.ReshapeExternalTensor(input_shape, input.base(), input_id)
        .ReshapeRuntime();
    ASSERT_EQ(subgraph.Status(), xnn_status_success);

    // Run subgraph without rewrites.
    Tensor<float> expected(subgraph.GetExternalTensorShape(output_id));
    subgraph.SetupExternalTensor(expected.base(), output_id)
        .SetupRuntime()
        .InvokeRuntime();

    // Create a runtime with rewrites.
    ASSERT_EQ(subgraph.CreateRuntime(), xnn_status_success);
    const size_t num_nodes_rewritten = subgraph.NumNodes();

    // Attach the inputs.
    subgraph.ReshapeExternalTensor(input_shape, input.base(), input_id)
        .ReshapeRuntime();
    ASSERT_EQ(subgraph.Status(), xnn_status_success);

    // Check the node type counts in the rewritten subgraph.
    for (const auto& entry : expected_node_type_counts) {
      const auto& node_type = entry.first;
      const auto& expected_count = entry.second;
      size_t count = 0;
      for (int k = 0; k < subgraph.NumNodes(); k++) {
        count += (subgraph.Node(k)->type == node_type);
      }
      ASSERT_EQ(count, expected_count)
          << "Unexpected number of " << xnn_node_type_to_string(node_type)
          << " nodes (expected " << expected_count << ", got " << count << ").";
    }

    // Run subgraph without rewrites.
    Tensor<float> output(subgraph.GetExternalTensorShape(output_id));
    subgraph.SetupExternalTensor(output.base(), output_id)
        .SetupRuntime()
        .InvokeRuntime();

    // Verify results.
    ASSERT_EQ(num_nodes_orig + expected_size_diff, num_nodes_rewritten);
    ASSERT_THAT(output, testing::ElementsAreArray(expected));
  }
}

class RewriteShapesTest : public ::testing::TestWithParam<int> {};

TEST_P(RewriteShapesTest, DoesNotElideLoadBearingNoOpReshape) {
  // Do not completely elide "load-bearing" nodes that are on the critical path
  // for the output, e.g. a single node between an input and an output value.
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input tensor to a different shape.
        std::vector<size_t> output_shape = input_shape.dims;
        std::shuffle(output_shape.begin(), output_shape.end(), rng);
        subgraph.AddReshape(output_shape, input_id, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteShapesTest, DoesNotElideLoadBearingReshape) {
  // Do not completely elide "load-bearing" nodes that are on the critical path
  // for the output, e.g. a single node between an input and an output value.
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input tensor to the same shape.
        subgraph.AddReshape(input_shape.dims, input_id, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteShapesTest, DoesNotElideLoadBearingExpandDims) {
  // Do not completely elide "load-bearing" nodes that are on the critical path
  // for the output, e.g. a single node between an input and an output value.
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input tensor to the same shape.
        subgraph.AddExpandDims({0}, input_id, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteShapesTest, ElidesReshapeOfStaticValue) {
  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Create a static value with a different shape from input1.
        uint32_t static_value_id = XNN_INVALID_VALUE_ID;
        std::vector<size_t> static_value_shape = input_shape.dims;
        std::shuffle(static_value_shape.begin(), static_value_shape.end(), rng);
        std::tie(static_tensor, static_value_id) =
            add_static_tensor<float>(rng, subgraph, static_value_shape);

        // Reshape the static tensor to the same shape as input.
        uint32_t reshaped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddReshape(input_shape.dims, static_value_id,
                            reshaped_value_id);

        // Add input1 and the reshaped static value, with the result going to
        // output1.
        auto inputs = random_swap(rng, reshaped_value_id, input_id);
        subgraph.AddAddition(inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/-1);
}

TEST_P(RewriteShapesTest, DoesNotElideNoOpReshapeOfNonStaticShapeValue) {
  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> extra_input_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Create a non-static value with a different shape from input1.
        extra_input_tensor = add_input_tensor<float>(rng, subgraph, input_shape,
                                                     extra_external_value_id);

        // Reshape the static tensor to the same shape as input.
        uint32_t reshaped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddReshape(input_shape.dims, extra_external_value_id,
                            reshaped_value_id);

        // Add input1 and the reshaped static value, with the result going to
        // output1.
        auto inputs = random_swap(rng, reshaped_value_id, input_id);
        subgraph.AddAddition(inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteShapesTest, DoesNotElidesReshapeOfStaticShapeNonStaticValue) {
  // Keep static tensor data in this scope so that it lives for the duration of
  // the test.
  Tensor<float> static_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Create a static value with a different shape from input1.
        uint32_t static_value_id = XNN_INVALID_VALUE_ID;
        std::vector<size_t> static_value_shape = input_shape.dims;
        std::shuffle(static_value_shape.begin(), static_value_shape.end(), rng);
        std::tie(static_tensor, static_value_id) =
            add_static_tensor<float>(rng, subgraph, static_value_shape);

        // Square the static tensor.
        uint32_t squared_value_id =
            add_internal_dynamic_tensor<float>(subgraph, static_value_shape);
        subgraph.AddUnary(xnn_unary_square, /*params=*/nullptr, static_value_id,
                          squared_value_id);

        // Reshape the squared value to the same shape as input.
        uint32_t reshaped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddReshape(input_shape.dims, squared_value_id,
                            reshaped_value_id);

        // Add input1 and the reshaped static value, with the result going to
        // output1.
        auto inputs = random_swap(rng, reshaped_value_id, input_id);
        subgraph.AddAddition(inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteShapesTest, CompactsASequenceOfReshapes) {
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input value.
        std::vector<size_t> shuffled_shape = input_shape.dims;
        std::shuffle(shuffled_shape.begin(), shuffled_shape.end(), rng);
        uint32_t reshaped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, shuffled_shape);
        subgraph.AddReshape(shuffled_shape, input_id, reshaped_value_id);

        // Reshape the input value again.
        std::shuffle(shuffled_shape.begin(), shuffled_shape.end(), rng);
        uint32_t rereshaped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, shuffled_shape);
        subgraph.AddReshape(shuffled_shape, reshaped_value_id,
                            rereshaped_value_id);

        // Reshape the input tensor back to its original shape.
        subgraph.AddReshape(input_shape.dims, rereshaped_value_id, output_id);
      },
      /*expected_size_diff=*/-2);
}

TEST_P(RewriteShapesTest, CompactsASequenceOfReshapesAndExpandDims) {
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input value.
        std::vector<size_t> shuffled_shape = input_shape.dims;
        std::shuffle(shuffled_shape.begin(), shuffled_shape.end(), rng);
        uint32_t reshaped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, shuffled_shape);
        subgraph.AddReshape(shuffled_shape, input_id, reshaped_value_id);

        // Expand the dims..
        std::vector<size_t> extra_dims =
            create_random_extra_dims(rng, input_shape.dims.size());
        uint32_t expanded_value_id = add_internal_dynamic_tensor<float>(
            subgraph, input_shape.dims.size() + extra_dims.size());
        subgraph.AddExpandDims(extra_dims, reshaped_value_id,
                               expanded_value_id);

        // Reshape the input tensor back to its original shape.
        subgraph.AddReshape(input_shape.dims, expanded_value_id, output_id);
      },
      /*expected_size_diff=*/-2);
}

TEST_P(RewriteShapesTest, CompactsASequenceOfExpandDimsAndReshape) {
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Expand the dims.
        std::vector<size_t> extra_dims =
            create_random_extra_dims(rng, input_shape.dims.size());
        uint32_t expanded_value_id = add_internal_dynamic_tensor<float>(
            subgraph, input_shape.dims.size() + extra_dims.size());
        subgraph.AddExpandDims(extra_dims, input_id, expanded_value_id);

        // Reshape the input value.
        std::vector<size_t> shuffled_shape = input_shape.dims;
        std::shuffle(shuffled_shape.begin(), shuffled_shape.end(), rng);
        uint32_t reshaped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, shuffled_shape);
        subgraph.AddReshape(shuffled_shape, expanded_value_id,
                            reshaped_value_id);

        // Reshape the input tensor back to its original shape.
        subgraph.AddReshape(input_shape.dims, reshaped_value_id, output_id);
      },
      /*expected_size_diff=*/-2);
}

class RewriteClampsTest : public ::testing::TestWithParam<int> {};

TEST_P(RewriteShapesTest, DoesNotElideLoadBearingNoOpClamp) {
  // Do not completely elide "load-bearing" nodes that are on the critical path
  // for the output, e.g. a single node between an input and an output value.
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        // Add a no-op clamp between the input and output.
        subgraph.AddClamp(-INFINITY, INFINITY, input_id, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteShapesTest, ElidesNoOpClamp) {
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a clamp in the range `[-INFINITY, INFINITY]`.
        uint32_t clamped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddClamp(-INFINITY, INFINITY, input_id, clamped_value_id);

        // Output the sum of the input and the clamped values.
        auto inputs = random_swap(rng, clamped_value_id, input_id);
        subgraph.AddAddition(inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/-1);
}

TEST_P(RewriteClampsTest, RewritesMinMaxWithStaticArgsToClamp) {
  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_min_tensor;
  Tensor<float> static_max_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add scalar static tensors for the min/max values.
        uint32_t static_min_value_id;
        uint32_t static_max_value_id;
        std::tie(static_min_tensor, static_min_value_id) =
            add_static_tensor<float>(rng, subgraph, {1});
        std::tie(static_max_tensor, static_max_value_id) =
            add_static_tensor<float>(rng, subgraph, {1});

        // Add the binary `minimum` op.
        uint32_t min_capped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        auto inputs = random_swap(rng, static_min_value_id, input_id);
        subgraph.AddBinary(xnn_binary_minimum, /*params=*/nullptr, inputs.first,
                           inputs.second, min_capped_value_id);

        // Add the binary `maximum` op.
        inputs = random_swap(rng, min_capped_value_id, static_max_value_id);
        subgraph.AddBinary(xnn_binary_maximum, /*params=*/nullptr, inputs.first,
                           inputs.second, output_id);
      },
      /*expected_size_diff=*/-1);
}

TEST_P(RewriteClampsTest, RewritesMaxMinWithStaticArgsToClamp) {
  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_min_tensor;
  Tensor<float> static_max_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add scalar static tensors for the min/max values.
        uint32_t static_min_value_id;
        uint32_t static_max_value_id;
        std::tie(static_min_tensor, static_min_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1});
        std::tie(static_max_tensor, static_max_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1});

        // Add the binary `maximum` op.
        uint32_t max_capped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        auto inputs = random_swap(rng, static_max_value_id, input_id);
        subgraph.AddBinary(xnn_binary_maximum, /*params=*/nullptr, inputs.first,
                           inputs.second, max_capped_value_id);

        // Add the binary `minimum` op.
        inputs = random_swap(rng, max_capped_value_id, static_min_value_id);
        subgraph.AddBinary(xnn_binary_minimum, /*params=*/nullptr, inputs.first,
                           inputs.second, output_id);
      },
      /*expected_size_diff=*/-1);
}

TEST_P(RewriteClampsTest, RewritesSequenceOfClampsToSingleClamps) {
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);
        std::uniform_real_distribution<float> rng_fp32(0.0, 0.5);

        // Add a first clamp in the range `[[0, 0.5), [0.5, 1)]`.
        uint32_t clamped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddClamp(rng_fp32(rng), 0.5 + rng_fp32(rng), input_id,
                          clamped_value_id);

        // Add a second clamp in the same range.
        subgraph.AddClamp(rng_fp32(rng), 0.5 + rng_fp32(rng), clamped_value_id,
                          output_id);
      },
      /*expected_size_diff=*/-1);
}

TEST_P(RewriteClampsTest,
       RewritesSequenceOfPotentiallyDisjointClampsToSingleClamps) {
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);
        std::uniform_real_distribution<float> rng_fp32(0.0, 2.0);

        // Add a first clamp in the range `[-1, 1]`.
        uint32_t clamped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddClamp(rng_fp32(rng) - 1.0, rng_fp32(rng) - 1.0, input_id,
                          clamped_value_id);

        // Add a second clamp in the same range.
        subgraph.AddClamp(rng_fp32(rng) - 1.0, rng_fp32(rng) - 1.0,
                          clamped_value_id, output_id);
      },
      /*expected_size_diff=*/-1);
}

TEST_P(RewriteClampsTest, DoesNotRewriteSharedSequenceOfClamps) {
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);
        std::uniform_real_distribution<float> rng_fp32(0.0, 0.5);

        // Add a first clamp in the range `[[0, 0.5), [0.5, 1)]`.
        uint32_t first_clamped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddClamp(rng_fp32(rng), 0.5 + rng_fp32(rng), input_id,
                          first_clamped_value_id);

        // Add a second clamp in the same range.
        uint32_t second_clamped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddClamp(rng_fp32(rng), 0.5 + rng_fp32(rng),
                          first_clamped_value_id, second_clamped_value_id);

        // Output the sum of the first and second clamped values.
        auto inputs =
            random_swap(rng, first_clamped_value_id, second_clamped_value_id);
        subgraph.AddAddition(inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/0);
}

class RewriteArithmeticTest : public ::testing::TestWithParam<int> {};

TEST_P(RewriteArithmeticTest, ElidesNoOpMul) {
  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_one_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a scalar static tensor with the value `1.0`.
        uint32_t static_one_value_id;
        std::tie(static_one_tensor, static_one_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 1.0, 1.0);

        // Add the binary `multiply` op with the constant 1.0.
        auto inputs = random_swap(rng, static_one_value_id, input_id);
        subgraph.AddBinary(xnn_binary_multiply, /*params=*/nullptr,
                           inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/0,
      /*expected_node_type_counts=*/
      {{xnn_node_type_copy, 1}, {xnn_node_type_binary_elementwise, 0}});
}

TEST_P(RewriteArithmeticTest, ElidesNoOpDiv) {
  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_one_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a scalar static tensor with the value `1.0`.
        uint32_t static_one_value_id;
        std::tie(static_one_tensor, static_one_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 1.0, 1.0);

        // Add the binary `divide` op with the constant 1.0.
        subgraph.AddBinary(xnn_binary_multiply, /*params=*/nullptr, input_id,
                           static_one_value_id, output_id);
      },
      /*expected_size_diff=*/0,
      /*expected_node_type_counts=*/
      {{xnn_node_type_copy, 1}, {xnn_node_type_binary_elementwise, 0}});
}

TEST_P(RewriteArithmeticTest, ElidesNoOpAdd) {
  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_zero_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a scalar static tensor with the value `0.0`.
        uint32_t static_zero_value_id;
        std::tie(static_zero_tensor, static_zero_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 0.0, 0.0);

        // Add the binary `add` op with the constant 0.0.
        auto inputs = random_swap(rng, static_zero_value_id, input_id);
        subgraph.AddBinary(xnn_binary_add, /*params=*/nullptr, inputs.first,
                           inputs.second, output_id);
      },
      /*expected_size_diff=*/0,
      /*expected_node_type_counts=*/
      {{xnn_node_type_copy, 1}, {xnn_node_type_binary_elementwise, 0}});
}

TEST_P(RewriteArithmeticTest, ElidesNoOpSub) {
  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_zero_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a scalar static tensor with the value `0.0`.
        uint32_t static_zero_value_id;
        std::tie(static_zero_tensor, static_zero_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 0.0, 0.0);

        // Add the binary `sub` op with the constant 0.0.
        subgraph.AddBinary(xnn_binary_subtract, /*params=*/nullptr, input_id,
                           static_zero_value_id, output_id);
      },
      /*expected_size_diff=*/0,
      /*expected_node_type_counts=*/
      {{xnn_node_type_copy, 1}, {xnn_node_type_binary_elementwise, 0}});
}

TEST_P(RewriteArithmeticTest, RewritesZeroMinusXToNegX) {
  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_zero_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a scalar static tensor with the value `0.0`.
        uint32_t static_zero_value_id;
        std::tie(static_zero_tensor, static_zero_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 0.0, 0.0);

        // Add the binary `sub` op with the constant 0.0.
        subgraph.AddBinary(xnn_binary_subtract, /*params=*/nullptr,
                           static_zero_value_id, input_id, output_id);
      },
      /*expected_size_diff=*/0,
      /*expected_node_type_counts=*/
      {{xnn_node_type_binary_elementwise, 0},
       {xnn_node_type_unary_elementwise, 1}});
}

TEST_P(RewriteArithmeticTest, ElidesNoOpChainOfMulZeroAdd) {
  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_zero_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a scalar static tensor with the value `0.0`.
        uint32_t static_zero_value_id;
        std::tie(static_zero_tensor, static_zero_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 0.0, 0.0);

        // Add the binary `multiply` op with the constant 0.0.
        uint32_t dynamic_zero_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        auto inputs = random_swap(rng, static_zero_value_id, input_id);
        subgraph.AddBinary(xnn_binary_multiply, /*params=*/nullptr,
                           inputs.first, inputs.second, dynamic_zero_value_id);

        // Add the binary `add` op with the input and the dynamic zero value.
        inputs = random_swap(rng, dynamic_zero_value_id, input_id);
        subgraph.AddBinary(xnn_binary_add, /*params=*/nullptr, inputs.first,
                           inputs.second, output_id);
      },
      /*expected_size_diff=*/-1,
      /*expected_node_type_counts=*/
      {{xnn_node_type_copy, 1}, {xnn_node_type_binary_elementwise, 0}});
}

TEST_P(RewriteArithmeticTest, ElidesNoOpChainOfDivOneMul) {
  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_one_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a scalar static tensor with the value `1.0`.
        uint32_t static_one_value_id;
        std::tie(static_one_tensor, static_one_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 1.0, 1.0);

        // Add the static `1.0` to the absolute value of the inputs to make sure
        // they are non-negative
        uint32_t abs_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        uint32_t shifted_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_abs, /*params=*/nullptr, input_id,
                          abs_value_id);
        auto inputs = random_swap(rng, abs_value_id, static_one_value_id);
        subgraph.AddAddition(inputs.first, inputs.second, shifted_value_id);

        // Add the binary `div(x, x)` op.
        uint32_t dynamic_one_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddBinary(xnn_binary_divide, /*params=*/nullptr,
                           shifted_value_id, shifted_value_id,
                           dynamic_one_value_id);

        // Add the binary `mul` op with the input and the dynamic one value.
        inputs = random_swap(rng, dynamic_one_value_id, input_id);
        subgraph.AddBinary(xnn_binary_multiply, /*params=*/nullptr,
                           inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/-3,
      /*expected_node_type_counts=*/
      {{xnn_node_type_copy, 1},
       {xnn_node_type_binary_elementwise, 0},
       {xnn_node_type_unary_elementwise, 0}});
}

TEST_P(RewriteArithmeticTest, RewritesAddOfNegValue) {
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        uint32_t exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_exp, /*params=*/nullptr, input_id,
                          exp_value_id);

        uint32_t neg_exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_negate, /*params=*/nullptr, exp_value_id,
                          neg_exp_value_id);

        // Add the negated value to the original input.
        auto inputs = random_swap(rng, neg_exp_value_id, input_id);
        subgraph.AddBinary(xnn_binary_add, /*params=*/nullptr, inputs.first,
                           inputs.second, output_id);
      },
      /*expected_size_diff=*/-1,
      /*expected_node_type_counts=*/
      {{xnn_node_type_binary_elementwise, 1},
       {xnn_node_type_unary_elementwise, 1}});
}

TEST_P(RewriteArithmeticTest, RewritesSubOfNegValue) {
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        uint32_t exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_exp, /*params=*/nullptr, input_id,
                          exp_value_id);

        uint32_t neg_exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_negate, /*params=*/nullptr, exp_value_id,
                          neg_exp_value_id);

        // Subgract the negated value from the original input.
        subgraph.AddBinary(xnn_binary_subtract, /*params=*/nullptr,
                           input_id, neg_exp_value_id, output_id);
      },
      /*expected_size_diff=*/-1,
      /*expected_node_type_counts=*/
      {{xnn_node_type_binary_elementwise, 1},
       {xnn_node_type_unary_elementwise, 1}});
}

TEST_P(RewriteArithmeticTest, RewritesMulOfNegValue) {
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        uint32_t exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_exp, /*params=*/nullptr, input_id,
                          exp_value_id);

        uint32_t neg_exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_negate, /*params=*/nullptr, exp_value_id,
                          neg_exp_value_id);

        uint32_t neg_input_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_negate, /*params=*/nullptr, input_id,
                          neg_input_value_id);

        // Multiply the two negated values.
        auto inputs = random_swap(rng, neg_exp_value_id, neg_input_value_id);
        subgraph.AddBinary(xnn_binary_multiply, /*params=*/nullptr,
                           inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/-2,
      /*expected_node_type_counts=*/
      {{xnn_node_type_binary_elementwise, 1},
       {xnn_node_type_unary_elementwise, 1}});
}

TEST_P(RewriteArithmeticTest, RewritesDivOfNegValue) {
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        uint32_t exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_exp, /*params=*/nullptr, input_id,
                          exp_value_id);

        uint32_t neg_exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_negate, /*params=*/nullptr, exp_value_id,
                          neg_exp_value_id);

        uint32_t neg_input_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_negate, /*params=*/nullptr, input_id,
                          neg_input_value_id);

        // Divide the two negated values.
        subgraph.AddBinary(xnn_binary_divide, /*params=*/nullptr,
                           neg_input_value_id, neg_exp_value_id, output_id);
      },
      /*expected_size_diff=*/-2,
      /*expected_node_type_counts=*/
      {{xnn_node_type_binary_elementwise, 1},
       {xnn_node_type_unary_elementwise, 1}});
}

INSTANTIATE_TEST_SUITE_P(Rewrite, RewriteShapesTest,
                         testing::Range(0, XNN_MAX_TENSOR_DIMS));
INSTANTIATE_TEST_SUITE_P(Rewrite, RewriteClampsTest, testing::Values(0, 1, 3));
INSTANTIATE_TEST_SUITE_P(Rewrite, RewriteArithmeticTest,
                         testing::Values(0, 1, 3));

}  // namespace xnnpack
