FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.docker/cli-plugins \
    && curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o /root/.docker/cli-plugins/docker-compose \
    && chmod +x /root/.docker/cli-plugins/docker-compose \
    && curl -fsSL https://ollama.com/install.sh | sh

ENV POETRY_VERSION=2.1.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    PATH="/opt/poetry/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -

RUN curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/nektos/act/master/install.sh | bash

FROM python:3.12-slim AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PATH="/opt/poetry/bin:$PATH"

COPY --from=builder /usr/bin/docker /usr/bin/docker
COPY --from=builder /opt/poetry /opt/poetry
COPY --from=builder /root/.docker/cli-plugins/docker-compose /root/.docker/cli-plugins/docker-compose
COPY --from=builder /usr/local/bin/ollama /usr/local/bin/ollama
COPY --from=builder /usr/bin/act /usr/bin/act


