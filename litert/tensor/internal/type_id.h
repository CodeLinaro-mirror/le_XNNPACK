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

#ifndef LITERT_TENSOR_INTERNAL_TYPE_ID_H_
#define LITERT_TENSOR_INTERNAL_TYPE_ID_H_

#include <cstdint>
#include <type_traits>

#include "absl/strings/string_view.h"

namespace litert::tensor::internal {

// Constexpr FNV-1a 64-bit hash function.
constexpr uint64_t ConstexprHash(absl::string_view str) {
  uint64_t hash = 14695981039346656037ull;
  for (char c : str) {
    hash ^= static_cast<uint64_t>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

class TypeId {
 public:
  template <typename T>
  static constexpr TypeId Get() {
    return GetExact<std::decay_t<T>>();
  }

  template <typename T>
  static constexpr TypeId GetExact() {
#if defined(__clang__) || defined(__GNUC__)
    constexpr absl::string_view name = __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
    constexpr absl::string_view name = __FUNCSIG__;
#else
#error "Unsupported compiler: function signature macro not available."
#endif
    return TypeId(ConstexprHash(name));
  }

  constexpr bool operator==(const TypeId& other) const {
    return id_ == other.id_;
  }
  constexpr bool operator!=(const TypeId& other) const {
    return id_ != other.id_;
  }

 private:
  explicit constexpr TypeId(uint64_t id) : id_(id) {}
  uint64_t id_;
};

}  // namespace litert::tensor::internal

#endif  // LITERT_TENSOR_INTERNAL_TYPE_ID_H_
