FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip ffmpeg libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN python3 -m pip install --upgrade pip &&     python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 &&     python3 -m pip install -r requirements.txt
COPY . .
RUN python3 -m pip install -e .
ENTRYPOINT ["python3", "-m", "moving_object_detection.cli"]
CMD ["--config", "configs/a100.yaml"]
