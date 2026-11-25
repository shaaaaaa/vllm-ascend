// SPDX-License-Identifier: Apache-2.0
#ifndef ASCEND_AIV_GATE_H
#define ASCEND_AIV_GATE_H

#include <acl/acl_rt.h>
#include <string.h>

// One foreground stream, one background stream, one aligned 64-bit device word.
// The word is the number of AIVs the background may use during a foreground
// batch. It describes an admission policy, not live hardware occupancy.
// The caller owns the word until the last replay using it finishes. Each
// concurrently executing graph instance needs its own word.
// An enqueue callback appends work to the supplied stream and returns its ACL
// status without throwing. Foreground batches are non-nested. A background
// callback appends exactly one bounded piece.
typedef aclError (*aiv_gate_enqueue_fn)(aclrtStream stream, void *work);

typedef struct {
    void *device_word;
    uint32_t capacity;
} aiv_gate;

// Enqueue this once before the streams fork. Use their usual event dependency
// so both streams observe the initialization. Each matched foreground batch
// restores the full capacity, allowing sequential replays to reuse the word.
static inline aclError aiv_gate_initialize(const aiv_gate *gate, aclrtStream stream)
{
    if (!gate || !gate->device_word || !gate->capacity) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return aclrtValueWrite(gate->device_word, gate->capacity, 0, stream);
}

// CMake checks the task-query declaration. Standalone users of this header
// can also match the graph-debug SDK or override the feature explicitly.
#ifndef AIV_GATE_HAS_TASK_PARAMS
#ifdef ACL_MDLRI_DEBUG_JSON_PRINT_SUMMARY
#define AIV_GATE_HAS_TASK_PARAMS 1
#else
#define AIV_GATE_HAS_TASK_PARAMS 0
#endif
#endif

#if AIV_GATE_HAS_TASK_PARAMS
// Upper bound from an already captured kernel. numBlocks describes a pure
// vector launch. Mixed, missing or opaque kernel metadata uses the full width.
// For a compound operation account for every child stream through its join.
static inline uint32_t aiv_gate_kernel_width(aclmdlRITask task, uint32_t capacity)
{
    aclmdlRITaskParams params;
    memset(&params, 0, sizeof(params));
    if (!task || aclmdlRITaskGetParams(task, &params) != ACL_SUCCESS ||
        params.type != ACL_MODEL_RI_TASK_KERNEL || !params.kernelTaskParams.funcHandle) {
        return capacity;
    }
    int64_t type = 0;
    if (aclrtGetFunctionAttribute(params.kernelTaskParams.funcHandle, ACL_FUNC_ATTR_KERNEL_TYPE,
                                 &type) != ACL_SUCCESS) {
        return capacity;
    }
    if (type == ACL_KERNEL_TYPE_CUBE || type == ACL_KERNEL_TYPE_AICPU) {
        return 0;
    }
    if (type != ACL_KERNEL_TYPE_VECTOR || !params.kernelTaskParams.numBlocks) {
        return capacity;
    }
    return params.kernelTaskParams.numBlocks < capacity ? params.kernelTaskParams.numBlocks : capacity;
}
#endif

// Submit contiguous foreground work independent of pieces it will block.
// Publish its AIV demand only after its inputs are ready. Restore the allowance
// before any join on background work needed by the foreground. Group adjacent
// work with the same bound; do not bracket individual blocks.
static inline aclError aiv_gate_foreground(const aiv_gate *gate, aclrtStream stream, uint32_t max_aiv,
                                          aiv_gate_enqueue_fn enqueue, void *work)
{
    if (!gate || !gate->device_word || !gate->capacity || max_aiv > gate->capacity || !enqueue) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (max_aiv == 0) {
        return enqueue(stream, work);
    }
    aclError status = aclrtValueWrite(gate->device_word, gate->capacity - max_aiv, 0, stream);
    if (status != ACL_SUCCESS) {
        return status;
    }
    status = enqueue(stream, work);
    // Preserve progress when the callback enqueued only part of the batch.
    aclError reopen_status = aclrtValueWrite(gate->device_word, gate->capacity, 0, stream);
    return reopen_status == ACL_SUCCESS ? status : reopen_status;
}

// The condition is evaluated by STARS on the device. One piece already
// admitted can finish after the foreground's smaller allowance becomes visible.
// max_aiv must upper-bound the physical AIVs the callback's single piece requests.
static inline aclError aiv_gate_background_piece(const aiv_gate *gate, aclrtStream stream, uint32_t max_aiv,
                                                aiv_gate_enqueue_fn enqueue, void *work)
{
    if (!gate || !gate->device_word || !max_aiv || max_aiv > gate->capacity || !enqueue) {
        return ACL_ERROR_INVALID_PARAM;
    }
    aclError status = aclrtValueWait(gate->device_word, max_aiv, ACL_STREAM_WAIT_VALUE_GEQ, stream);
    if (status != ACL_SUCCESS) {
        return status;
    }
    return enqueue(stream, work);
}

#endif
