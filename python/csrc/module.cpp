/**
 * @file module.cpp
 * @brief ASPL Python 绑定主模块
 * 
 * 使用 pybind11 将 C++ 接口暴露给 Python
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

// TODO: 添加 Python 绑定
// 示例：
// #include <aspl/ops/ops.h>
// py::module m = py::module::create_extension_module("aspl", "ASPL Python bindings");
// py::class_<aspl::ops::Reduction>(m, "Reduction")...

PYBIND11_MODULE(aspl, m) {
    m.doc() = "ASPL: AI System Performance Lab Python bindings";
    
    // TODO: 添加模块绑定
}

