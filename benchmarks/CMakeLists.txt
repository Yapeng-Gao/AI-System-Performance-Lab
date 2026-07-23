cmake_minimum_required(VERSION 3.25)

# 先确保 CUDA Toolkit 目标可用（提供 CUDA::cudart 等）
find_package(CUDAToolkit REQUIRED)

# --- 独立微基准（不依赖 NVBench）---
# 说明：上游 NVIDIA/nvbench 的 main 现已要求 CMake >= 3.30.4。
# 本目录三个 .cu 本身只用 CUDA Runtime，与 Toolkit/Driver 是否 13.2 无关。
# 因此默认不拉取 nvbench，避免把项目 cmake 门槛抬到 3.30。

add_executable(bench_attention_memory_bound attention_memory_bound.cu)
add_executable(bench_cp_async_pipeline      cp_async_pipeline.cu)
add_executable(bench_hbm_pointer_chasing    hbm_pointer_chasing.cu)

target_link_libraries(bench_attention_memory_bound PRIVATE CUDA::cudart)
target_link_libraries(bench_cp_async_pipeline      PRIVATE CUDA::cudart)
target_link_libraries(bench_hbm_pointer_chasing    PRIVATE CUDA::cudart)

set_target_properties(
        bench_attention_memory_bound
        bench_cp_async_pipeline
        bench_hbm_pointer_chasing
        PROPERTIES CXX_STANDARD 17
)

# --- 可选：NVBench（需要本机 CMake >= 3.30.4）---
# 仓库当前没有依赖 nvbench 的源文件；仅在显式打开且 cmake 足够新时拉取，便于以后扩展。
option(ASPL_ENABLE_NVBENCH "Fetch NVIDIA nvbench (requires CMake >= 3.30.4)" OFF)
if(ASPL_ENABLE_NVBENCH)
    if(CMAKE_VERSION VERSION_LESS 3.30.4)
        message(FATAL_ERROR
            "ASPL_ENABLE_NVBENCH=ON needs CMake >= 3.30.4 (found ${CMAKE_VERSION}). "
            "Upgrade cmake, or leave ASPL_ENABLE_NVBENCH=OFF (default).")
    endif()

    include(FetchContent)
    if(WIN32)
        set(NVBench_ENABLE_CUPTI OFF CACHE BOOL "Disable CUPTI on Windows" FORCE)
        set(NVBench_ENABLE_NVML  OFF CACHE BOOL "Disable NVML on Windows"  FORCE)
        add_compile_options(
                $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
                $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        )
    endif()

    FetchContent_Declare(
            nvbench
            GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
            GIT_TAG        main
            GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(nvbench)
    message(STATUS "NVBench enabled (optional). Add a bench_* target that links nvbench when needed.")
endif()
