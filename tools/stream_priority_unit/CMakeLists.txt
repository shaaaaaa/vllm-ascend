# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(stream_priority_unit LANGUAGES C CXX)
set(RUN_MODE npu CACHE STRING "Ascend C execution mode")
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
ascendc_library(stream_priority_pulse SHARED pulse.cpp)
