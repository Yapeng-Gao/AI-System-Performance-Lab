# 针对不同架构的编译选项

# 检测 CUDA 版本（要求 12.0 以上）
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
    message(FATAL_ERROR "CUDA version ${CMAKE_CUDA_COMPILER_VERSION} < 12.0. This project requires CUDA 12.0 or higher.")
endif()

# 基础 Flags
set(CUDA_BASE_FLAGS
        "--use_fast_math"
        "--expt-relaxed-constexpr"
        "--expt-extended-lambda"
        "-Xcompiler=/wd4819"  # Windows下忽略特定警告
        "-Xptxas=-v"          # 输出寄存器使用情况
        "-lineinfo"           # Nsight 分析支持
)

# 针对架构优化
# 80=A100, 86=RTX3090, 89=RTX4090, 90=H100
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

# 只有当 aspl_core 目标存在时才应用 flags
if(TARGET aspl_core)
    target_compile_options(aspl_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_BASE_FLAGS}>)
    message(STATUS "Applied NVCC flags to aspl_core")
endif()