cmake_minimum_required(VERSION 3.25)

# --- 1. 引入 NVBench ---
include(FetchContent)

FetchContent_Declare(
        nvbench
        GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
        GIT_TAG        main
        GIT_SHALLOW    TRUE
)
# 这一步会自动下载并配置 NVBench
FetchContent_MakeAvailable(nvbench)

# --- 2. 定义 Benchmark 可执行文件 ---
# 注意：把所有 .cu 基准加入构建目标中
add_executable(bench_flash_attn bench_flash_attn.cu)
add_executable(bench_attention_memory_bound attention_memory_bound.cu)
add_executable(bench_cp_async_pipeline    cp_async_pipeline.cu)
add_executable(bench_hbm_pointer_chasing  hbm_pointer_chasing.cu)

# --- 3. 链接依赖 ---
# 链接 NVBench（仅 flash_attn 用到）和 CUDA 运行时
target_link_libraries(bench_flash_attn            PRIVATE nvbench CUDA::cudart)
target_link_libraries(bench_attention_memory_bound PRIVATE CUDA::cudart)
target_link_libraries(bench_cp_async_pipeline      PRIVATE CUDA::cudart)
target_link_libraries(bench_hbm_pointer_chasing    PRIVATE CUDA::cudart)

# --- 4. 设置编译属性 ---
# NVBench 强烈依赖 C++17；其他可执行也统一为 C++17
set_target_properties(
        bench_flash_attn
        bench_attention_memory_bound
        bench_cp_async_pipeline
        bench_hbm_pointer_chasing
        PROPERTIES CXX_STANDARD 17
)

# 针对 CUDA 文件的特殊标志 (允许扩展 lambda 等特性)
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(bench_flash_attn PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
    )
endif()