###########################
#= STAGE 1: BUILD STAGE =#
##########################

FROM python:3.10.12-alpine AS builder

# SHELL [ "/bin/bash", "-c" ]

ENV VIRTUAL_ENV=/opt/venv
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

## Install build dependencies
RUN apk update && \
    apk add --no-cache gcc libffi-dev libressl-dev vim && \
    rm -rf /var/cache/apk/*

## Install `uv` for fast package management
COPY --from=ghcr.io/astral-sh/uv:python3.10-12-alpine /uv /uvx /bin/
RUN uv venv ${VIRTUAL_ENV}

## Set the PATH to use the virtual environment
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
COPY Codes/Python/requirements.txt /tmp/
RUN uv pip install --no-cache-dir -r /tmp/requirements.txt

############################
#= STAGE 2: RUNTIME STAGE =#
############################

FROM python:3.10.12-alpine AS runtime
COPY --from=builder /opt/venv /opt/venv

## Metadata
LABEL author="A. Mhamdi <a_mhamdi@outlook.com>"
LABEL version="latest"

ENV VIRTUAL_ENV=/opt/venv
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

WORKDIR /workspace
COPY Codes/Python/Datasets/ Datasets/
COPY Codes/Python/Jupyter/ Jupyter/
COPY Codes/Python/Marimo/ Marimo/

EXPOSE 1357 2468

CMD [ "/bin/bash" ]

