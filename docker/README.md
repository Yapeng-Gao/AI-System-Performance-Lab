# Docker 环境

本目录包含 Docker 配置文件和容器化环境。

## 计划内容

- `Dockerfile` - CUDA 开发环境镜像
- `Dockerfile.dev` - 开发环境（包含所有工具）
- `docker-compose.yml` - 多容器编排
- `.dockerignore` - Docker 构建忽略文件

## 使用说明

```bash
# 构建镜像
docker build -t aspl:latest -f docker/Dockerfile .

# 运行容器
docker run --gpus all -it aspl:latest

# 使用 docker-compose
docker-compose up
```

