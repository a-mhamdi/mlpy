###########################
#= STAGE 1: BUILD STAGE =#
##########################

FROM python:3.10.12 AS builder

SHELL [ "/bin/bash", "-c" ]

ENV VIRTUAL_ENV=/opt/venv
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

## Install build dependencies
RUN apt-get update && \
	apt-get install -y --no-install-recommends gcc libffi-dev libssl-dev vim && \
    rm -rf /var/lib/apt/lists/*

## Install `uv` for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv venv ${VIRTUAL_ENV}

## Set the PATH to use the virtual environment
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
COPY ../Codes/Python/requirements.txt /tmp/
RUN uv pip install --no-cache-dir -r /tmp/requirements.txt

############################
#= STAGE 2: RUNTIME STAGE =#
############################

FROM python:3.10.12 AS runtime
COPY --from=builder /opt/venv /opt/venv

## Metadata
LABEL author="A. Mhamdi <a_mhamdi@outlook.com>"
LABEL version="latest"

ENV VIRTUAL_ENV=/opt/venv
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

WORKDIR /workspace
COPY ../Codes/Python/datasets/ datasets/

EXPOSE 1357 2468

CMD [ "/bin/bash" ]

