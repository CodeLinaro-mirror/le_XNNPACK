// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>

#include <gtest/gtest.h>


class AttentionOperatorTester {
 public:
  inline AttentionOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline AttentionOperatorTester& cap(float cap) {
    this->cap_ = cap;
    return *this;
  }

  inline float cap() const {
    return this->cap_;
  }

  inline AttentionOperatorTester& sequence_length(size_t sequence_length) {
    this->sequence_length_ = sequence_length;
    return *this;
  }

  inline size_t sequence_length() const {
    return this->sequence_length_;
  }

  inline AttentionOperatorTester& channels(size_t channels) {
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline AttentionOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scaledist(0.2f, 2.0f);

    std::vector<float> query(XNN_EXTRA_BYTES / sizeof(float) + batch_size() * sequence_length() * channels());
    std::vector<float> key(XNN_EXTRA_BYTES / sizeof(float) + batch_size() * sequence_length() * channels());
    std::vector<float> value(XNN_EXTRA_BYTES / sizeof(float) + batch_size() * sequence_length() * channels());
    // Use a different distribution to avoid divide by 0.
    std::vector<float> scale(XNN_EXTRA_BYTES / sizeof(float) + channels());
    std::vector<float> mask(XNN_EXTRA_BYTES / sizeof(float) + sequence_length() * sequence_length());
    std::vector<float> output(batch_size() * sequence_length() * channels());
    std::vector<float> output_ref(batch_size() * sequence_length() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(query.begin(), query.end(), [&]() { return f32dist(rng); });
      std::generate(scale.begin(), scale.end(), [&]() { return scaledist(rng); });
      std::generate(key.begin(), key.end(), [&]() { return f32dist(rng); });
      std::generate(value.begin(), value.end(), [&]() { return f32dist(rng); });
      std::generate(mask.begin(), mask.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      for (size_t b = 0; b < batch_size(); b++) {
        // Compute reference results.
        std::vector<float> q_scaled(sequence_length() * channels());
        for (size_t n = 0; n < sequence_length(); n++) {
          for (size_t k = 0; k < channels(); k++) {
            q_scaled[n * channels() + k] =
                query[b * sequence_length() * channels() + n * channels() + k] * scale[k];
          }
        }

        std::vector<float> logits(sequence_length() * sequence_length(), 0);
        for (size_t n_0 = 0; n_0 < sequence_length(); n_0++) {
          for (size_t n_1 = 0; n_1 < sequence_length(); n_1++) {
            for (size_t ki = 0; ki < channels(); ki++) {
              logits[n_0 * sequence_length() + n_1] +=
                  q_scaled[n_0 * channels() + ki] *
                  key[b * sequence_length() * channels() + n_1 * channels() + ki];
            }
            if (cap() > 0.0f) {
              // Cap and tanh.
              logits[n_0 * sequence_length() + n_1] =
                  std::tanh(logits[n_0 * sequence_length() + n_1] / cap()) * cap();
            }
            // Mask.
            logits[n_0 * sequence_length() + n_1] += mask[n_0 * sequence_length() + n_1];
          }
        }

        std::vector<float> weights(sequence_length() * sequence_length(), 0.0f);

        for (size_t i = 0; i < sequence_length(); i++) {
          // Online softmax per row.
          float mv = -std::numeric_limits<double>::infinity();
          float dv = 0;
          for (size_t j = 0; j < sequence_length(); j++) {
            float prev_m = mv;
            mv = std::max(prev_m, logits[i * sequence_length() + j]);
            dv = dv * exp(prev_m - mv) + exp(logits[i * sequence_length() + j] - mv);
          }
          for (size_t j = 0; j < sequence_length(); j++) {
            weights[i * sequence_length() + j] = exp(logits[i * sequence_length() + j] - mv)/ dv;
          }
        }

        // Output = Weights * Value
        for (size_t ni = 0; ni < sequence_length(); ni++) {
          for (size_t nj = 0; nj < sequence_length(); nj++) {
            for (size_t di = 0; di < channels(); di++) {
              output_ref[b * sequence_length() * channels() + ni * channels() + di] +=
                  weights[ni * sequence_length() + nj] *
                  value[b * sequence_length() * channels() + nj * channels() + di];
            }
          }
        }
      }

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t attention_op = nullptr;
      const xnn_status status = xnn_create_scaled_dot_attention_nhtc_f32(
          /*flags=*/0,
          &attention_op);

      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(attention_op, nullptr);

      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_attention_op(attention_op, xnn_delete_operator);

      size_t workspace_size = 0;
      size_t workspace_alignment = 0;
      ASSERT_EQ(xnn_status_success,
                xnn_reshape_scaled_dot_attention_nhtc_f32(
                  attention_op,
                  batch_size(),
                  sequence_length(),
                  channels(),
                  cap(),
                  &workspace_size, &workspace_alignment,
                  /*threadpool=*/nullptr));

      ASSERT_NE(workspace_size, 0);
      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size, 0);

      ASSERT_EQ(xnn_status_success,
                xnn_setup_scaled_dot_attention_nhtc_f32(
                  attention_op, workspace.data(), query.data(), key.data(), value.data(), scale.data(), mask.data(), output.data()));

      ASSERT_EQ(xnn_status_success, xnn_run_operator(attention_op, /*threadpool=*/nullptr));

      for (size_t b = 0; b < batch_size(); b++) {
        for (size_t i = 0; i < sequence_length(); i++) {
          for (size_t j = 0; j < channels(); j++) {
            EXPECT_NEAR(output_ref[b * sequence_length() * channels() + i * channels() + j],
                        output[b * sequence_length() * channels() + i * channels() + j],
                        1e-2)
                << "batch : " << b << " / "  << batch_size()
                << " sequence length : " << i << " / " << sequence_length()
                << " channel : " << j << " / " << channels();
          }
        }
      }
    }
  }

 private:
  float cap_{0.0f};
  size_t batch_size_{1};
  size_t channels_{1};
  size_t sequence_length_{1};
  size_t iterations_{1};
};
