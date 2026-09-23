set(RESIDENT_BASE_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../csrc/kernels/resident_sorted_cache.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${RESIDENT_BASE_SOURCE}")
file(READ "${RESIDENT_BASE_SOURCE}" resident_source)
foreach(kernel sharded_union sorted_read_probe sorted_finalize sorted_update sorted_state_update sorted_remap)
    string(REPLACE "dsa_resident_${kernel}_kernel" "dsa_resident_${kernel}_exact_kernel" resident_source "${resident_source}")
endforeach()
string(REPLACE "namespace vllm_ascend {" "namespace vllm_ascend { namespace resident_exact {" resident_source "${resident_source}")
# The additional namespace encloses host launchers only.
set(RESIDENT_EXACT_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/resident_sorted_cache_exact.cpp")
file(WRITE "${RESIDENT_EXACT_SOURCE}"
"#define RESIDENT_EXPERIMENT_VECTOR_INTERSECTION 1
#define RESIDENT_EXPERIMENT_VECTOR_STATE 1
${resident_source}
} // namespace vllm_ascend
")
