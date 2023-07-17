// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "attention-operator-tester.h"

TEST(ATTENTION_NHTC_F32, tiny_size_1x1) {
  AttentionOperatorTester()
    .batch_size(1)
    .sequence_length(1)
    .channels(1)
    .TestF32();
}

TEST(ATTENTION_NHTC_F32, mid_size_105x137) {
  AttentionOperatorTester()
      .batch_size(3)
      .sequence_length(105)
      .channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, mid_size_105x137_with_cap) {
  AttentionOperatorTester()
      .batch_size(3)
      .cap(50.0f)
      .sequence_length(105)
      .channels(137)
      .TestF32();
}

// Sizes used by BERT.
// https://github.com/google-research/bert
TEST(ATTENTION_NHTC_F32, bert_sizes) {
  std::vector<size_t> batch_sizes = {1, 2};
  std::vector<size_t> sequence_lengths = {128, 384, 512};
  std::vector<size_t> channels = {128, 256, 512, 768};
  for (size_t b : batch_sizes) {
    for (size_t s : sequence_lengths) {
      for (size_t c : channels) {
        AttentionOperatorTester()
          .batch_size(b)
          .sequence_length(s)
          .channels(c)
          .TestF32();
      }
    }
  }
}

// https://huggingface.co/transformers/v3.3.1/pretrained_models.html
TEST(ATTENTION_NHTC_F32, gpt2_sizes) {
  std::vector<size_t> sequence_lengths = {128};
  std::vector<size_t> channels = {768, 1024, 1280, 1600};
  for (size_t s : sequence_lengths) {
    for (size_t c : channels) {
      AttentionOperatorTester()
        .sequence_length(s)
        .channels(c)
        .TestF32();
    }
  }
}