# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(wse_persistent_sim LANGUAGES CXX)

get_filename_component(SIMPLER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(PERSISTENT_RUNTIME "${SIMPLER_ROOT}/src/wse/runtime/persistent")
add_library(wse_persistent STATIC device.cpp "${PERSISTENT_RUNTIME}/decode.cpp"
    "${PERSISTENT_RUNTIME}/admission.cpp" "${PERSISTENT_RUNTIME}/prefill_state.cpp")
target_include_directories(wse_persistent PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${PERSISTENT_RUNTIME}"
    "${SIMPLER_ROOT}/src/wse/platform/include")
target_compile_features(wse_persistent PUBLIC cxx_std_17)
target_compile_options(wse_persistent PRIVATE -Wall -Wextra -Wpedantic)

add_executable(wse_decode_sim "${SIMPLER_ROOT}/examples/wse/persistent_decode/decode_sim.cpp"
    "${SIMPLER_ROOT}/examples/wse/persistent_decode/live_input.cpp")
target_link_libraries(wse_decode_sim PRIVATE wse_persistent)

include(CTest)
if(BUILD_TESTING)
    add_executable(wse_persistent_tests "${SIMPLER_ROOT}/tests/ut/cpp/wse/persistent/device_test.cpp")
    target_link_libraries(wse_persistent_tests PRIVATE wse_persistent)
    add_test(NAME wse_persistent_protocol COMMAND wse_persistent_tests)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy"
        RESULT_VARIABLE NUMPY_STATUS ERROR_QUIET)
    if(NOT NUMPY_STATUS EQUAL 0)
        message(FATAL_ERROR "The test interpreter needs NumPy; use the project-local .venv")
    endif()
    add_test(NAME wse_persistent_decode COMMAND "${Python3_EXECUTABLE}"
        "${SIMPLER_ROOT}/examples/wse/persistent_decode/run_checks.py"
        --binary "$<TARGET_FILE:wse_decode_sim>" --output "${CMAKE_BINARY_DIR}/validation")
    set_tests_properties(wse_persistent_decode PROPERTIES TIMEOUT 180)
    add_executable(wse_admission_tests "${SIMPLER_ROOT}/tests/ut/cpp/wse/persistent/admission_test.cpp")
    target_link_libraries(wse_admission_tests PRIVATE wse_persistent)
    add_test(NAME wse_persistent_live_admission COMMAND "${Python3_EXECUTABLE}"
        "${SIMPLER_ROOT}/examples/wse/persistent_decode/run_live_checks.py"
        --binary "$<TARGET_FILE:wse_decode_sim>" --api-binary "$<TARGET_FILE:wse_admission_tests>"
        --output "${CMAKE_BINARY_DIR}/live-validation")
    set_tests_properties(wse_persistent_live_admission PROPERTIES TIMEOUT 180)
endif()
