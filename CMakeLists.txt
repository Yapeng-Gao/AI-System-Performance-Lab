cmake_minimum_required(VERSION 4.0)

project(AI_System_Performance_Lab LANGUAGES C CXX CUDA)


# --- 1. 全局配置 ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 设置模块路径
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# 输出目录设置 (避免 Windows 下乱跑)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- 2. 核心库 (ASPL Core) 定义 ---
include_directories(include)

# 自动扫描源码
file(GLOB_RECURSE ASPL_SRCS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp")
file(GLOB_RECURSE ASPL_KERNELS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/*.cu")

# 确保至少有一个源文件（添加 dummy.cpp 作为占位）
if(NOT ASPL_SRCS)
    list(APPEND ASPL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/dummy.cpp")
endif()

# 必！须！是 SHARED 库，不是可执行文件
add_library(aspl_core SHARED ${ASPL_SRCS} ${ASPL_KERNELS})

# 链接 CUDA
target_link_libraries(aspl_core PUBLIC cuda cudart)

# 统一输出目录（放在 add_library 后面即可）
# Windows: .dll 和 .exe 都放在 bin，.lib 放在 lib
# Linux: .so 放在 lib，可执行文件放在 bin
if(WIN32)
    set_target_properties(aspl_core PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib   # *.lib
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin   # *.dll
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin   # *.exe
    )
else()
    set_target_properties(aspl_core PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib   # *.a
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib   # *.so
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin   # 可执行文件
    )
endif()

# --- 3. 配置 Flags (修复顺序问题) ---
# 注意：必须在 add_library 之后 include 这个文件，因为里面用到了 target_compile_options(aspl_core ...)
include(NVCCFlags)

# --- 4. 构建子模块 ---
# 只有当目录存在且含有 CMakeLists.txt 时才添加
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
else()
    message(WARNING "examples/CMakeLists.txt not found, skipping examples build.")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    option(BUILD_TESTS "Build unit tests" OFF)
    if(BUILD_TESTS)
        add_subdirectory(tests)
    endif()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
    option(BUILD_BENCHMARKS "Build benchmarks" OFF)
    if(BUILD_BENCHMARKS)
        add_subdirectory(benchmarks)
    endif()
endif()