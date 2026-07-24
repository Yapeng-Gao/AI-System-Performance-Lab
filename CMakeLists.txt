cmake_minimum_required(VERSION 3.25)

project(AI_System_Performance_Lab LANGUAGES C CXX CUDA)

# --- 1. 全局配置 ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- 2. NVCC / 架构 ---
include(NVCCFlags)

# --- 3. 专栏示例（主产物）---
# 仓库已瘦身为 article/ + examples/ 主线；不再构建占位 aspl_core / benchmarks / tests。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
else()
    message(FATAL_ERROR "examples/CMakeLists.txt not found")
endif()
