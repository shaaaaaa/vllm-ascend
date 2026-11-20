# SPDX-License-Identifier: Apache-2.0
set(sparse_flash_attention_dcp_packet_depends sparse_flash_attention PARENT_SCOPE)
target_sources(op_host_aclnn PRIVATE sparse_flash_attention_dcp_packet_def.cpp)
add_ops_compile_options(
    OP_NAME SparseFlashAttentionDcpPacket
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Werror -fpermissive
)
