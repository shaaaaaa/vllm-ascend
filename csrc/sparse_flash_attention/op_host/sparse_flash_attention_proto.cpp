/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
constexpr size_t QUERY_INPUT_INDEX = 0;

ge::graphStatus InferShapeSparseFlashAttention(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("SparseFlashAttention", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)
    *attentionOutShape = *queryShape;
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseFlashAttention(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("SparseFlashAttention", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SparseFlashAttention).InferShape(InferShapeSparseFlashAttention).InferDataType(InferDataTypeSparseFlashAttention);

ge::graphStatus InferShapeSparseFlashAttentionDcp(gert::InferShapeContext *context)
{
    if (InferShapeSparseFlashAttention(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const auto *query = context->GetInputShape(QUERY_INPUT_INDEX);
    if (query->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 1; i <= 2; ++i) {
        auto *shape = context->GetOutputShape(i);
        if (shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        shape->SetDimNum(2);
        shape->SetDim(0, query->GetDim(0));
        shape->SetDim(1, query->GetDim(1));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseFlashAttentionDcp(gert::InferDataTypeContext *context)
{
    if (InferDataTypeSparseFlashAttention(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SparseFlashAttentionDcp)
    .InferShape(InferShapeSparseFlashAttentionDcp)
    .InferDataType(InferDataTypeSparseFlashAttentionDcp);
} // namespace ops
  
