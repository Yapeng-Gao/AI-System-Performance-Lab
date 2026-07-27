# 针对不同架构的编译选项

# 检测 CUDA 版本（要求 12.0 以上）
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
    message(FATAL_ERROR "CUDA version ${CMAKE_CUDA_COMPILER_VERSION} < 12.0. This project requires CUDA 12.0 or higher.")
endif()

# 基础 Flags（供需要时复用；examples 目标自带 -lineinfo / -Xptxas=-v）
if(WIN32)
    set(CUDA_BASE_FLAGS
            "--use_fast_math"
            "--expt-relaxed-constexpr"
            "--expt-extended-lambda"
            "-Xcompiler=/wd4819"
            "-Xptxas=-v"
            "-lineinfo"
    )
else()
    set(CUDA_BASE_FLAGS
            "--use_fast_math"
            "--expt-relaxed-constexpr"
            "--expt-extended-lambda"
            "-Xptxas=-v"
            "-lineinfo"
    )
endif()

# 针对架构优化
# 80=A100, 86=RTX3090, 89=RTX4090, 90=H100/Hopper TMA；120=Blackwell（RTX 5090）
# 本机可传 -DCMAKE_CUDA_ARCHITECTURES=native / 120 等覆盖默认
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
    # B-08 TMA 等 sm_90+ 特性在 Blackwell 上需要 sm_120 SASS（或依赖 PTX JIT）
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
        list(APPEND CMAKE_CUDA_ARCHITECTURES "120")
    endif()
endif()
