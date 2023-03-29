// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdint.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
      size_t g,                                              \
      size_t nc,                                             \
      size_t kc,                                             \
      size_t nr,                                             \
      const uint32_t* b,                                     \
      uint32_t* packed_weights,                              \
      size_t extra_bytes,                                    \
      const union xnn_x32_packb_params* params);             \

DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_x2__scalar_float)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_x2__scalar_int)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_x4__scalar_float)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_x4__scalar_int)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_x8__scalar_float)
DECLARE_X32_PACKB_GEMM_UKERNEL_FUNCTION(xnn_x32_packb_gemm_ukernel_x8__scalar_int)

#ifdef __cplusplus
}  // extern "C"
#endif
