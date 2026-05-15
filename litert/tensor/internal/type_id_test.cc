/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "litert/tensor/internal/type_id.h"

#include <gtest/gtest.h>

namespace litert::tensor::internal {
namespace {

template <int N>
struct DummyTemplate {};

struct DummyA {};
struct DummyB {};

TEST(TypeIdTest, Uniqueness) {
  // Verify distinct concrete types have distinct TypeIds.
  EXPECT_NE(TypeId::Get<DummyA>(), TypeId::Get<DummyB>());
  EXPECT_NE(TypeId::Get<int>(), TypeId::Get<float>());

  // Verify template specializations are precisely disambiguated.
  // If the compiler macro stops including template arguments, these checks will
  // fail.
  EXPECT_NE(TypeId::Get<DummyTemplate<1>>(), TypeId::Get<DummyTemplate<2>>());
  EXPECT_NE(TypeId::Get<DummyTemplate<100>>(),
            TypeId::Get<DummyTemplate<200>>());
}

TEST(TypeIdTest, Equality) {
  EXPECT_EQ(TypeId::Get<DummyA>(), TypeId::Get<DummyA>());
  EXPECT_EQ(TypeId::Get<DummyTemplate<1>>(), TypeId::Get<DummyTemplate<1>>());
}

TEST(TypeIdTest, Decay) {
  // Verify const and reference types decay transparently.
  EXPECT_EQ(TypeId::Get<const DummyA>(), TypeId::Get<DummyA>());
  EXPECT_EQ(TypeId::Get<DummyA&>(), TypeId::Get<DummyA>());
}

}  // namespace
}  // namespace litert::tensor::internal
