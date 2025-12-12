# 第三方依赖

本目录用于存放第三方库的源代码（如果需要作为子模块或本地依赖）。

## 计划内容

- 可能包含一些需要在本地编译的第三方库
- 使用 Git Submodules 管理的依赖

## 使用说明

```bash
# 初始化子模块
git submodule update --init --recursive

# 更新子模块
git submodule update --remote
```

## 注意事项

- 大多数依赖通过 CMake 的 `FetchContent` 或 `find_package` 管理
- 只有特殊情况才需要将第三方代码放在本目录

