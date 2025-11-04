###########################
#= STAGE 1: BUILD STAGE =#
##########################

FROM python:3.10.12-bookworm-slim AS builder

# SHELL [ "/bin/bash", "-c" ]

ENV VIRTUAL_ENV=/opt/venv
    PYTHONDONTWRITEBYTECODE=1
    PYTHONUNBUFFERED=1
    PATH="${VIRTUAL_ENV}/bin:$PATH"

# Install build dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    gcc \
    libffi-dev \
    libssl-dev \
    vim && \
    rm -rf /var/lib/apt/lists/*

## Install `uv` for fast package management
COPY --from=ghcr.io/astral-sh/uv:python3.10-bookworm-slim /uv /uvx /bin/

# Make the binaries executable
RUN chmod +x /bin/uv /bin/uvx

RUN uv venv ${VIRTUAL_ENV}

## Set the PATH to use the virtual environment
COPY Codes/Python/requirements.txt /tmp/
RUN uv pip install --no-cache-dir -r /tmp/requirements.txt

RUN apt-get update && \
    apt-get purge --auto-remove -y gcc libffi-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*
    
############################
#= STAGE 2: RUNTIME STAGE =#
############################

FROM python:3.10.12-bookworm-slim AS runtime
COPY --from=builder /opt/venv /opt/venv

## Metadata
LABEL author="A. Mhamdi <a_mhamdi@outlook.com>"
LABEL version="latest"

ENV VIRTUAL_ENV=/opt/venv
    PYTHONDONTWRITEBYTECODE=1
    PYTHONUNBUFFERED=1
    PATH="${VIRTUAL_ENV}/bin:$PATH"

WORKDIR /workspace
COPY Codes/Python/Datasets/ Datasets/
COPY Codes/Python/Jupyter/ Jupyter/
COPY Codes/Python/Marimo/ Marimo/

EXPOSE 1357 2468

CMD [ "/bin/bash" ]

