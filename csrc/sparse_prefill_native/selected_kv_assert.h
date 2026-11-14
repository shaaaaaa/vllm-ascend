// SPDX-License-Identifier: Apache-2.0
#pragma once

// These kernels rely on device assertions for invalid union/status/address
// inputs. A build that disables them must fail, rather than emit a kernel
// that can silently return with incomplete KV before the following SFA.
// Check before the first SDK include, where the assertion macros are defined.
#ifdef NDEBUG
#error "Selected-KV kernels require assertions in Release: compile with -UNDEBUG"
#endif
#if !defined(ASCENDC_DUMP) || ASCENDC_DUMP != 1
#error "Selected-KV kernels require ASCENDC_DUMP=1"
#endif

#include "kernel_operator.h"

#ifndef ASSERT_MSG
#error "Selected-KV kernels require the SDK device ASSERT_MSG API"
#endif
