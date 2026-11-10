set(RESIDENT_BASE_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../csrc/kernels/resident_sorted_cache.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${RESIDENT_BASE_SOURCE}")
file(READ "${RESIDENT_BASE_SOURCE}" resident_source)

function(generate_resident_variant variant output)
    set(source "${resident_source}")
    foreach(kernel sharded_union sorted_read_probe sorted_finalize sorted_update sorted_state_update sorted_remap)
        string(REPLACE "dsa_resident_${kernel}_kernel"
            "dsa_resident_${kernel}_${variant}_kernel" source "${source}")
    endforeach()
    # The additional namespace encloses host launchers only.
    string(REPLACE "namespace vllm_ascend {"
        "namespace vllm_ascend { namespace resident_${variant} {" source "${source}")
    set(defines "#define RESIDENT_EXPERIMENT_VECTOR_INTERSECTION 1\n#define RESIDENT_EXPERIMENT_VECTOR_STATE 1\n")
    if(variant STREQUAL "membership" OR variant STREQUAL "dcp")
        string(APPEND defines "#define RESIDENT_EXPERIMENT_QUERY_MEMBERSHIP 1\n")
    endif()
    if(variant STREQUAL "dcp")
        string(APPEND defines "#define RESIDENT_DCP_SHARDING 1\n")
        string(REPLACE "dsa_resident_sorted_membership_update_kernel"
            "dsa_resident_sorted_membership_update_dcp_kernel" source "${source}")
    endif()
    set(path "${CMAKE_CURRENT_BINARY_DIR}/resident_sorted_cache_${variant}.cpp")
    file(WRITE "${path}" "${defines}${source}\n} // namespace vllm_ascend\n")
    set(${output} "${path}" PARENT_SCOPE)
endfunction()

generate_resident_variant(exact RESIDENT_EXACT_SOURCE)
generate_resident_variant(membership RESIDENT_MEMBERSHIP_SOURCE)
generate_resident_variant(dcp RESIDENT_DCP_SOURCE)
