# Base image
FROM python:3.10.12 AS builder

# Metadata
LABEL author="A. Mhamdi"
LABEL version="latest"

SHELL [ "/bin/bash", "-c" ]

ENV VIRTUAL_ENV=/opt/venv
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY COPY ./Codes/Python/requirements.txt .

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Install `uv` for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv venv ${VIRTUAL_ENV}

# Set the PATH to use the virtual environment
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
RUN uv pip install pip --upgrade
RUN uv pip install --no-cache-dir -r requirements.txt

FROM python:3.10.12 AS runtime
COPY --from=builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Activate `venv`
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

WORKDIR /app
COPY COPY ./Codes/Python/datasets/ datasets/

CMD [ "/bin/bash" ]
