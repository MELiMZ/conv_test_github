# 使用 NVIDIA CUDA 镜像作为基础
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# 更新包索引并安装 Python 和 pip
RUN apt-get update && apt-get install -y python3 python3-pip

# 安装 TensorFlow 和其他必要的 Python 包
RUN pip3 install tensorflow

# 复制应用程序代码到容器中
COPY . /app

# 设置工作目录
WORKDIR /app

# 定义容器启动时运行的命令
CMD ["python3", "train.py"]