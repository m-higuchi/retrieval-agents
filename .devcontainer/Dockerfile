FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
#    libffi-dev \
    build-essential \
    make \
    g++ \
    cmake \
    python3-dev \
    git \
    openssh-client \
    docker.io curl ca-certificates \
    libmagic1 \
    && mkdir -p /root/.docker/cli-plugins \
    && curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
         -o /root/.docker/cli-plugins/docker-compose \
    && chmod +x /root/.docker/cli-plugins/docker-compose \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip