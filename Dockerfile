# 使用官方带 CUDA 支持的 PyTorch 基础镜像
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# 设置工作目录
WORKDIR /workspace

# 拷贝当前目录下所有文件到容器
COPY . .

# 安装 Python 包（可拓展）
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision

# 默认执行 test_conv3d_mem.py
CMD ["python", "test_conv3d_mem.py"]
