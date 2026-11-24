# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(stream_priority_unit LANGUAGES C CXX)
set(RUN_MODE npu CACHE STRING "Ascend C execution mode")
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
ascendc_fatbin_library(stream_priority_pulse pulse.cpp)
add_library(stream_priority_host SHARED submit.cpp)
target_include_directories(stream_priority_host PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_libraries(stream_priority_host PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64/libacl_rt.so")
