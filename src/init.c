// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <unistd.h>
#endif

#ifdef _MSC_VER
  #include <intrin.h>
#endif

#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/params.h"

#if XNN_ENABLE_KLEIDIAI
#include <stdio.h>

#include "kai/kai_common.h"
#endif  // XNN_ENABLE_KLEIDIAI

XNN_INIT_ONCE_GUARD(allocator);

static const struct xnn_allocator* volatile init_allocator = NULL;

static void init_allocator_config(void) {
  uint32_t init_flags = XNN_INIT_FLAG_XNNPACK;
  memcpy(&xnn_params.allocator, init_allocator, sizeof(struct xnn_allocator));
  xnn_params.init_flags = init_flags;
}

enum xnn_status xnn_initialize(const struct xnn_allocator* allocator) {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    xnn_log_error("XNNPACK initialization failed: hardware not supported");
    return xnn_status_unsupported_hardware;
  }

  if (allocator == NULL) {
    allocator = &xnn_default_allocator;
  }
  #ifdef _MSC_VER
    _InterlockedCompareExchangePointer((PVOID volatile*) &init_allocator, (PVOID) allocator, NULL);
  #else
    __sync_bool_compare_and_swap(&init_allocator, NULL, allocator);
  #endif
  XNN_INIT_ONCE(allocator);
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) != 0) {
    return xnn_status_success;
  } else {
    return xnn_status_unsupported_hardware;
  }

#if XNN_ENABLE_KLEIDIAI
  /* If we're using KleidiAI, log their version string. */
  static bool first = true;
  if (first) {
    first = false;
    fprintf(stderr,
            "XNNPACK built with KleidiAI v%s (see "
            "https://gitlab.arm.com/kleidi/kleidiai).\n",
            kai_get_version());
  }
#endif  // XNN_ENABLE_KLEIDIAI
}

enum xnn_status xnn_deinitialize(void) {
  return xnn_status_success;
}

// Initialization guards keep track of the config generation they were
// initialized on.
//
// If the generation stored in the guard is different from this value it means
// the initialization needs to be run again.
//
// This is initialized to 1 to invalidate guards that are initialized to 0.
uint32_t xnn_init_generation = 1;

#if XNN_PLATFORM_WINDOWS || XNN_HAS_PTHREADS
void xnn_init_once_impl(struct xnn_init_guard* guard, XNN_ONCE_LOCK_TYPE* lock, void (*init_fn)(void)) {
#if XNN_PLATFORM_WINDOWS
  AcquireSRWLockExclusive(lock);
#elif XNN_HAS_PTHREADS
  pthread_mutex_lock(lock);
#endif

  if (guard->generation != xnn_init_generation) {
    init_fn();
    guard->generation = xnn_init_generation;
  }

#if XNN_PLATFORM_WINDOWS
  ReleaseSRWLockExclusive(lock);
#elif XNN_HAS_PTHREADS
  pthread_mutex_unlock(lock);
#endif
}
#else
void xnn_init_once_impl(struct xnn_init_guard* guard, void (*init_fn)(void)) {
  if (guard->generation != xnn_init_generation) {
    init_fn();
    guard->generation = xnn_init_generation;
  }
}
#endif

void xnn_reset_all_configs(void) {
  xnn_init_generation++;
  if (xnn_init_generation == 0) {
    xnn_init_generation = 1;
  }
}
