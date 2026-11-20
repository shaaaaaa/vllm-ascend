# SPDX-License-Identifier: Apache-2.0
set(SELECTED_KV_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(_asc_tgts sparse_prefill_union_pipeline sparse_prefill_fused_fixed dsa_bf16_mla_inplace)

ascendc_library(sparse_prefill_union_pipeline SHARED "${CMAKE_CURRENT_LIST_DIR}/sparse_prefill_union_pipeline.cpp")
ascendc_library(sparse_prefill_fused_fixed SHARED "${CMAKE_CURRENT_LIST_DIR}/sparse_prefill_fused_fixed.cpp")
ascendc_library(dsa_bf16_mla_inplace SHARED "${CMAKE_CURRENT_LIST_DIR}/bf16_mla_inplace.cpp")

foreach(tgt IN ITEMS ${_asc_tgts})
    ascendc_compile_definitions(${tgt} PRIVATE ASCENDC_DUMP=1 ONE_CORE_DUMP_SIZE=1024)
    ascendc_compile_options(${tgt} PRIVATE
        -UNDEBUG -forward-options-to-host-compiler -UNDEBUG -DASCENDC_DUMP=1 -DONE_CORE_DUMP_SIZE=1024
    )
endforeach()

add_custom_command(TARGET sparse_prefill_fused_fixed POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SELECTED_KV_LIBRARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_BINARY_DIR}/sparse_prefill_fused_fixed_merge_obj_dir/device_aiv.o"
        "${SELECTED_KV_LIBRARY_DIR}/sparse_prefill_fused_fixed.o"
)

add_library(sparse_prefill_fused_fixed_launch SHARED "${CMAKE_CURRENT_LIST_DIR}/sparse_prefill_batch_queued.cpp")
target_include_directories(sparse_prefill_fused_fixed_launch PRIVATE
    "${ASCEND_HOME_PATH}/include" ${TORCH_INCLUDE_DIRS} "${TORCH_NPU_PATH}/include"
)
target_link_directories(sparse_prefill_fused_fixed_launch PRIVATE "${ASCEND_HOME_PATH}/lib64" "${TORCH_NPU_PATH}/lib")
target_link_libraries(sparse_prefill_fused_fixed_launch PRIVATE
    ascendcl sparse_prefill_union_pipeline ${TORCH_LIBRARIES} torch_npu
)
add_dependencies(sparse_prefill_fused_fixed_launch sparse_prefill_fused_fixed)
separate_arguments(torch_abi_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(sparse_prefill_fused_fixed_launch PRIVATE ${torch_abi_flags})
target_link_options(sparse_prefill_fused_fixed_launch PRIVATE "-Wl,-rpath,$ORIGIN")

set(SELECTED_KV_TARGETS ${_asc_tgts} sparse_prefill_fused_fixed_launch)
set_target_properties(${SELECTED_KV_TARGETS} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${SELECTED_KV_LIBRARY_DIR}" BUILD_RPATH_USE_ORIGIN TRUE INSTALL_RPATH "$ORIGIN"
)
add_dependencies(vllm_ascend_C ${SELECTED_KV_TARGETS})

# Prefer explicit VLLM_ASCEND_PACKAGE_PATH when provided; otherwise also cover wheel build_lib because
# untouched setup.py:296 instantiates `develop` even during bdist_wheel builds and skips copying lib/*.o.
if(DEFINED VLLM_ASCEND_PACKAGE_PATH)
    set(_selected_dirs "${VLLM_ASCEND_PACKAGE_PATH}/lib")
else()
    string(REGEX REPLACE "[/\\\\]vllm_ascend$" "" _selected_pkg_root "${VLLM_ASCEND_INSTALL_PATH}")
    get_filename_component(_bld_parent "${CMAKE_BINARY_DIR}" DIRECTORY)
    string(REGEX REPLACE "[/\\\\]temp\\.([^/\\\\]+)$" "/lib.\\1/vllm_ascend" _wheel_pkg "${CMAKE_BINARY_DIR}")
    set(_selected_dirs "${_selected_pkg_root}/vllm_ascend/lib")
    foreach(_cand IN ITEMS "${CMAKE_SOURCE_DIR}/vllm_ascend" "${_wheel_pkg}" "${_bld_parent}/lib/vllm_ascend")
        if(EXISTS "${_cand}")
            list(APPEND _selected_dirs "${_cand}/lib")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _selected_dirs)
endif()

foreach(_dir IN LISTS _selected_dirs)
    install(TARGETS ${SELECTED_KV_TARGETS} LIBRARY DESTINATION "${_dir}")
    install(FILES "${SELECTED_KV_LIBRARY_DIR}/sparse_prefill_fused_fixed.o" DESTINATION "${_dir}")
endforeach()
