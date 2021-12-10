#pragma once

#include <cstddef>

namespace xnnpack {
constexpr size_t kCodeSizeInBytes = 131072;  // 128kb.

struct xnn_code_buffer {
  // Pointer to allocated, externally managed memory.
  void* code;
  // Actual size of instructions (bytes). It is only safe to access code within this size.
  size_t size;
  // Maximum capacity of the buffer pointer to by `code`. This is the size of the currently mapped memory.
  size_t capacity;
};

struct xnn_code_buffer jit_alloc(size_t size = kCodeSizeInBytes);
void jit_finalize(xnn_code_buffer* buf);
void jit_free(xnn_code_buffer& buf);
}  // namespace xnnpack
