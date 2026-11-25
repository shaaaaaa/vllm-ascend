# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(stream_priority_unit LANGUAGES C CXX)
set(RUN_MODE npu CACHE STRING "Ascend C execution mode")
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
ascendc_fatbin_library(stream_priority_pulse pulse.cpp)
add_library(stream_priority_host SHARED submit.cpp)
option(BUILD_AIV_GATE_PROBE "Include the device-side AIV admission test" OFF)
if(BUILD_AIV_GATE_PROBE)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES "${ASCEND_CANN_PACKAGE_PATH}/include")
    check_cxx_source_compiles([[
        #include <acl/acl_rt.h>
        int main() { (void)sizeof(&aclmdlRITaskGetParams); }
    ]] AIV_GATE_SDK_HAS_TASK_PARAMS)
    target_compile_definitions(stream_priority_host PRIVATE
        AIV_GATE_HAS_TASK_PARAMS=$<BOOL:${AIV_GATE_SDK_HAS_TASK_PARAMS}>)
    target_sources(stream_priority_host PRIVATE gate_probe.cpp)
endif()
target_include_directories(stream_priority_host PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_libraries(stream_priority_host PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64/libacl_rt.so")
