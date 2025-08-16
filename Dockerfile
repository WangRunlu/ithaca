FROM python:3.8-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=on PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates unzip tar git build-essential ffmpeg libgl1 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && pip install -r /workspace/requirements.txt
ENV DATAROOT=/data/sets/ithaca365
COPY . /workspace
CMD ["/bin/bash"]
