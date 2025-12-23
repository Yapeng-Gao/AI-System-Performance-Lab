# examples/CMakeLists.txt
# 自动化构建脚本：自动扫描所有子目录下的 .cu/.cpp 文件并生成独立的可执行程序

# --- 新增：查找 CUDA 工具包 (用于 NVRTC 和 Driver API) ---
find_package(CUDAToolkit REQUIRED)

# 1. 定义宏：如何构建一个 CUDA/C++ 示例
macro(create_cuda_example target_name source_file)
    # 确保源文件路径是绝对路径
    if(IS_ABSOLUTE ${source_file})
        set(abs_source_file ${source_file})
    else()
        get_filename_component(abs_source_file ${source_file} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
    
    # 明确设置 .cu 文件的语言属性为 CUDA（必须在 add_executable 之前）
    if(abs_source_file MATCHES "\\.cu$")
        set_source_files_properties(${abs_source_file} PROPERTIES LANGUAGE CUDA)
    endif()
    
    add_executable(${target_name} ${abs_source_file})

    # 读取源代码内容用于依赖检测
    file(READ ${abs_source_file} source_content)

    # --- 逻辑 A: 检测是否需要链接 aspl_core ---
    string(FIND "${source_content}" "aspl/" found_aspl_header)
    string(FIND "${source_content}" "#include.*aspl" found_aspl_include)

    if(found_aspl_header GREATER_EQUAL 0 OR found_aspl_include GREATER_EQUAL 0)
        target_link_libraries(${target_name} PRIVATE aspl_core)
        message(STATUS "[Example] ${target_name}: Linked with aspl_core")
    else()
        # 基础 CUDA 运行时 (使用 Modern CMake Target)
        target_link_libraries(${target_name} PRIVATE CUDA::cudart)
    endif()

    # --- 逻辑 B: 检测是否需要 NVRTC / Driver API (第06章需求) ---
    string(FIND "${source_content}" "nvrtc.h" found_nvrtc)
    string(FIND "${source_content}" "cuda.h" found_driver) # Driver API 通常包含 cuda.h

    if(found_nvrtc GREATER_EQUAL 0 OR found_driver GREATER_EQUAL 0)
        target_link_libraries(${target_name} PRIVATE
                CUDA::nvrtc
                CUDA::cuda_driver
        )
        message(STATUS "[Example] ${target_name}: Linked with NVRTC & Driver API")
    endif()

    # --- 逻辑 C: 编译选项配置 ---
    # 启用 CUDA 分离编译 (仅针对 .cu 文件)
    if(abs_source_file MATCHES "\\.cu$")
        set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()

    # 添加编译器选项 (行号信息用于 Nsight 分析)
    if(WIN32)
        # Windows: 抑制字符编码相关警告（C4819 等），通过 MSVC 风格的 /wd4819
        target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xptxas=-v -Xcompiler=/wd4819>
                $<$<COMPILE_LANGUAGE:CXX>:/wd4819>
        )
    else()
        # Linux/macOS: 不使用 /wd4819，避免被 gcc/clang 当成“额外输入文件”
        target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xptxas=-v>
        )
    endif()

    message(STATUS "[Example] Added target: ${target_name}")
endmacro()

# 2. 自动扫描：递归查找所有 .cu 和 .cpp 文件 (新增 .cpp 支持)
file(GLOB_RECURSE ALL_EXAMPLE_SRCS 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu" 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

foreach(source_file ${ALL_EXAMPLE_SRCS})
    # 解析文件路径
    # 例如 source_file = .../examples/01_cuda_basics/06_nvrtc_jit.cpp

    # 获取文件名 (不含扩展名) -> 06_nvrtc_jit
    get_filename_component(file_name ${source_file} NAME_WE)

    # 获取父目录名 -> 01_cuda_basics
    get_filename_component(dir_path ${source_file} DIRECTORY)
    get_filename_component(dir_name ${dir_path} NAME)

    # 组合成唯一的 Target 名称 -> 01_cuda_basics_06_nvrtc_jit
    set(target_name "${dir_name}_${file_name}")

    # 创建目标
    create_cuda_example(${target_name} ${source_file})

endforeach()