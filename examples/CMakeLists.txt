# examples/CMakeLists.txt
# 自动化构建脚本：自动扫描所有子目录下的 .cu 文件并生成独立的可执行程序

# 1. 定义宏：如何构建一个 CUDA 示例
macro(create_cuda_example target_name source_file)
    add_executable(${target_name} ${source_file})

    # 检查源文件是否包含 aspl 头文件，决定是否需要链接 aspl_core
    file(READ ${source_file} source_content)
    string(FIND "${source_content}" "aspl/" found_aspl_header)
    string(FIND "${source_content}" "#include.*aspl" found_aspl_include)
    
    # 如果找到了 aspl 相关引用，则链接 aspl_core
    if(found_aspl_header GREATER_EQUAL 0 OR found_aspl_include GREATER_EQUAL 0)
        target_link_libraries(${target_name} PRIVATE aspl_core)
        message(STATUS "[Example] ${target_name}: Linked with aspl_core")
    else()
        # 只链接 CUDA 运行时（标准 CUDA 程序）
        target_link_libraries(${target_name} PRIVATE cuda cudart)
        message(STATUS "[Example] ${target_name}: Standalone CUDA program (no aspl_core)")
    endif()

    # 启用 CUDA 分离编译 (支持 __device__ 函数跨文件调用)
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # 添加编译器选项 (行号信息用于 Nsight 分析)
    # 抑制 Windows 下的字符编码警告（C4819 和 Unicode 相关警告）
    target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xptxas=-v -Xcompiler=/wd4819>
            $<$<COMPILE_LANGUAGE:CXX>:/wd4819>
    )

    message(STATUS "[Example] Added target: ${target_name}")
endmacro()

# 2. 自动扫描：递归查找所有 .cu 文件
file(GLOB_RECURSE ALL_EXAMPLE_SRCS "*.cu")

foreach(source_file ${ALL_EXAMPLE_SRCS})
    # 解析文件路径
    # 例如 source_file = .../examples/01_cuda_basics/01_hello_modern.cu

    # 获取文件名 (不含扩展名) -> 01_hello_modern
    get_filename_component(file_name ${source_file} NAME_WE)

    # 获取父目录名 -> 01_cuda_basics
    get_filename_component(dir_path ${source_file} DIRECTORY)
    get_filename_component(dir_name ${dir_path} NAME)

    # 组合成唯一的 Target 名称 -> 01_cuda_basics_01_hello_modern
    # 这样即使不同目录下都有 main.cu 也不会冲突
    set(target_name "${dir_name}_${file_name}")

    # 创建目标
    create_cuda_example(${target_name} ${source_file})

endforeach()