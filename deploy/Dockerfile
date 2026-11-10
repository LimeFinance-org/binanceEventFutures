# GPU 镜像（需要宿主机安装 nvidia-container-toolkit）
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/becai
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
COPY . .
EXPOSE 8600
CMD ["python3", "-m", "becai.cli", "web", "--host", "0.0.0.0"]
