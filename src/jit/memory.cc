#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>

#include "xnnpack/jit-memory.h"

namespace xnnpack {
struct xnn_code_buffer jit_alloc(size_t size) {
  void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return {p, 0, size};
}

void jit_finalize(xnn_code_buffer* buf) {
  // Make code RX only.
  mprotect(buf->code, buf->size, PROT_READ | PROT_EXEC);

  // Release all unused pages.
  long page_size = sysconf(_SC_PAGE_SIZE);
  if (page_size < 0) {
    // Can't retrieve page size, don't unmap anything.
    return;
  }
  // Round up used memory to page-aligned size.
  long page_aligned = ~(page_size - 1);
  size_t page_aligned_code_size = (size_t)(buf->size + page_size) & page_aligned;
  uint8_t* addr = (uint8_t*)buf->code + page_aligned_code_size;
  if (addr >= buf->code) {
    // We used up every page. No need to unmap anything.
    return;
  }
  size_t new_capacity = buf->capacity - page_aligned_code_size;
  munmap((void*)addr, new_capacity);
  buf->capacity = new_capacity;
}

void jit_free(xnn_code_buffer& buf) {
  munmap(buf.code, buf.size);
}
}  // namespace xnnpack
