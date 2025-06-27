# Dockerfile
FROM registry.cn-beijing.aliyuncs.com/acs-sample/python:3.9-slim

WORKDIR /app

COPY app.py .

ENTRYPOINT ["python", "app.py"]
