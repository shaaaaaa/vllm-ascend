# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from unittest.mock import patch

import torch

from vllm_ascend.worker.npu_input_batch import NPUInputBatch


def _make_input_batch(layerwise_prefill: bool) -> NPUInputBatch:
    with patch(
        "vllm_ascend.worker.npu_input_batch.envs_ascend."
        "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE",
        layerwise_prefill,
    ):
        return NPUInputBatch(
            max_num_reqs=2,
            max_model_len=8,
            max_num_batched_tokens=8,
            device=torch.device("cpu"),
            pin_memory=False,
            vocab_size=32,
            block_sizes=[2, 4],
            kernel_block_sizes=[[2], [4]],
            max_num_blocks_per_req=[4, 2],
        )


def test_ordinary_input_batch_reuses_primary_block_table() -> None:
    input_batch = _make_input_batch(layerwise_prefill=False)

    assert input_batch.layerwise_prefill_block_tables == (
        input_batch.block_table,
    )


def test_layerwise_prefill_input_batch_owns_three_matching_tables() -> None:
    input_batch = _make_input_batch(layerwise_prefill=True)
    tables = input_batch.layerwise_prefill_block_tables

    assert len(tables) == 3
    assert tables[0] is input_batch.block_table
    assert len({id(table) for table in tables}) == 3
    assert [len(table.block_tables) for table in tables] == [2, 2, 2]

    tables[1].add_row(([10, 11], [20]), row_idx=0)
    assert tables[1][0].num_blocks_per_row[0] == 2
    assert tables[1][1].num_blocks_per_row[0] == 1
    assert tables[0][0].num_blocks_per_row[0] == 0
    assert tables[2][0].num_blocks_per_row[0] == 0
