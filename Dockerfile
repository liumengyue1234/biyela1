FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制 Python 依赖文件
COPY models/requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制模型代码
COPY models/ .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/checkpoints
ENV UPLOAD_PATH=/app/uploads

# 创建必要的目录
RUN mkdir -p $MODEL_PATH $UPLOAD_PATH

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "scripts/run_detection.py"]
