# SPDX-License-Identifier: Apache-2.0
# Device kernels for selected-KV prefill. These are built only on the A3
# target and stay separate from the ordinary operator library.
set(SELECTED_KV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
ascendc_library(sparse_prefill_union_pipeline SHARED "${SELECTED_KV_SOURCE_DIR}/sparse_prefill_union_pipeline.cpp")
ascendc_library(sparse_prefill_fused_fixed SHARED "${SELECTED_KV_SOURCE_DIR}/sparse_prefill_fused_fixed.cpp")
ascendc_library(dsa_packed_inplace SHARED "${SELECTED_KV_SOURCE_DIR}/packed_inplace.cpp")
ascendc_library(dsa_selected_li_pages SHARED "${SELECTED_KV_SOURCE_DIR}/li_page_gather.cpp")

# Address/status checks are required for correctness, including Release builds.
# The host launcher reserves one 1024-byte dump slot per core, so pin the
# generated host stub to the same per-core size instead of relying on its
# default.
foreach(selected_kv_target IN ITEMS sparse_prefill_union_pipeline sparse_prefill_fused_fixed dsa_packed_inplace)
    ascendc_compile_definitions(${selected_kv_target} PRIVATE ASCENDC_DUMP=1 ONE_CORE_DUMP_SIZE=1024)
    ascendc_compile_options(${selected_kv_target} PRIVATE
        -UNDEBUG
        -forward-options-to-host-compiler -UNDEBUG -DASCENDC_DUMP=1 -DONE_CORE_DUMP_SIZE=1024)
endforeach()

# Keep the tested main-kernel build settings unchanged; LI is a separate target.
ascendc_compile_definitions(dsa_selected_li_pages PRIVATE ASCENDC_DUMP=1)
ascendc_compile_options(dsa_selected_li_pages PRIVATE
    -UNDEBUG
    -forward-options-to-host-compiler -UNDEBUG -DASCENDC_DUMP=1)

set(SELECTED_KV_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib")
if(NOT DEFINED VLLM_ASCEND_PACKAGE_PATH)
    set(VLLM_ASCEND_PACKAGE_PATH "${VLLM_ASCEND_INSTALL_PATH}/vllm_ascend")
endif()

# The generated host .so is a build artifact; the runtime asset is the AIV
# object loaded through aclrtBinaryLoadFromFile().
add_custom_command(TARGET sparse_prefill_fused_fixed POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SELECTED_KV_LIBRARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_BINARY_DIR}/sparse_prefill_fused_fixed_merge_obj_dir/device_aiv.o"
        "${SELECTED_KV_LIBRARY_DIR}/sparse_prefill_fused_fixed.o")

# Host launch/queue wrapper. It loads the fixed kernel object, runs the
# prefix union entrypoint from sparse_prefill_union_pipeline, then launches
# sparse_prefill_partition_finish_fixed with the target's 12-argument
# signature. It also registers the LI page-gather queue so the installed
# package exports torch.ops.dsa_prefill_queue.li_page_gather.
add_library(sparse_prefill_fused_fixed_launch SHARED
    "${SELECTED_KV_SOURCE_DIR}/sparse_prefill_fused_fixed_launch.cpp"
    "${SELECTED_KV_SOURCE_DIR}/sparse_prefill_batch_queued.cpp"
    "${SELECTED_KV_SOURCE_DIR}/li_page_gather_queued.cpp"
    "${SELECTED_KV_SOURCE_DIR}/hybm_address_export.cpp")
target_include_directories(sparse_prefill_fused_fixed_launch PRIVATE
    "${ASCEND_HOME_PATH}/include" ${TORCH_INCLUDE_DIRS} "${TORCH_NPU_PATH}/include")
target_link_directories(sparse_prefill_fused_fixed_launch PRIVATE
    "${ASCEND_HOME_PATH}/lib64" "${TORCH_NPU_PATH}/lib")
target_link_libraries(sparse_prefill_fused_fixed_launch PRIVATE
    ascendcl sparse_prefill_union_pipeline dsa_selected_li_pages ${TORCH_LIBRARIES} torch_npu)
add_dependencies(sparse_prefill_fused_fixed_launch sparse_prefill_fused_fixed dsa_selected_li_pages)
separate_arguments(torch_abi_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(sparse_prefill_fused_fixed_launch PRIVATE ${torch_abi_flags})

set(SELECTED_KV_TARGETS
    sparse_prefill_union_pipeline
    sparse_prefill_fused_fixed
    sparse_prefill_fused_fixed_launch
    dsa_packed_inplace
    dsa_selected_li_pages)
set_target_properties(${SELECTED_KV_TARGETS} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${SELECTED_KV_LIBRARY_DIR}"
    BUILD_RPATH_USE_ORIGIN TRUE INSTALL_RPATH "$ORIGIN")
add_dependencies(vllm_ascend_C ${SELECTED_KV_TARGETS})
install(TARGETS ${SELECTED_KV_TARGETS} LIBRARY DESTINATION "${VLLM_ASCEND_PACKAGE_PATH}/lib")
install(FILES "${SELECTED_KV_LIBRARY_DIR}/sparse_prefill_fused_fixed.o"
    DESTINATION "${VLLM_ASCEND_PACKAGE_PATH}/lib")

# The host wrapper owns the selected runtime queue registration, including
# torch.ops.dsa_prefill_queue.li_page_gather. The Python selected runtime and
# token-count host rounding are still unwired; the launcher validates only
# static geometry and device counts stay on device.
