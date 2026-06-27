FROM nvidia/cuda:13.3.0-cudnn-devel-ubuntu22.04

ARG user=dev

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y \
    git \
    curl ca-certificates \
    sudo \
    wget \
    libcusparselt0 libcusparselt-dev


# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Setup python env with uv
RUN mkdir /fold
ADD docker/download_bert.py /fold
ADD pyproject.toml /fold
ADD uv.lock /fold
ADD .python-version /fold

WORKDIR /fold
# Install dependencies only; skip building the local package so this
# layer is cached even when source files (including README.md) haven't
# been added yet.  The project itself is built on first `uv run` after
# ADD . /fold below.
RUN uv sync --no-install-project
# cache download of bert
RUN uv run --no-project download_bert.py

ADD . /fold
