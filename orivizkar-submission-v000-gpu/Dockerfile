FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp/origami-home \
    XDG_CACHE_HOME=/tmp/origami-cache \
    IPYTHONDIR=/tmp/origami-ipython \
    TORCH_HOME=/tmp/origami-cache/torch \
    TORCHINDUCTOR_CACHE_DIR=/tmp/origami-cache/torchinductor \
    TRITON_CACHE_DIR=/tmp/origami-cache/triton

# Install python3-dev (provides Python.h) along with gcc and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.4 wheels and dependencies
# Must run BEFORE the lock install: the lock leaves torch unpinned, so without
# this the default PyPI wheel (cu130) lands in this cuda:12.4.1 base image and
# fails at runtime on any driver older than 580.
# NOTE: no --extra-index-url here. pip picks the highest version across all
# indexes, so adding PyPI lets its cu130 wheels outrank every cu124 wheel and
# silently defeats the pin. Versions are explicit because cu124 caps at these.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
    --default-timeout=100 \
    --retries=5 \
    --index-url "${TORCH_INDEX_URL}" \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124
    # --no-cache-dir

COPY requirements-inference.lock /tmp/requirements-inference.lock
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
    -r /tmp/requirements-inference.lock \
    && rm /tmp/requirements-inference.lock
    # --no-cache-dir
    #to remove mount after final img is done, replace with no cache dir

# Copy source code and entrypoint
COPY vitacformer /app/vitacformer
COPY vitac_policy_server.py /app/vitac_policy_server.py
COPY smoothing.py /app/smoothing.py
COPY entrypoint.sh /app/entrypoint.sh

# Copy the checkpoint set. A training run is one self-contained directory --
# exactly what origami_imitate_episodes.py writes, e.g.
#   checkpoints/aug20_resnet18_224_jitter_obsfps5_20260820_065144_ori_tactile/
#       policy_epoch_70_loss_0.027.ckpt      (possibly several epochs)
#       training_configs.json                 (architecture: hidden_dim, backbone,
#                                              camera order, mask_fingers, ...)
#       normalizer_config.json                (normalization stats)
#       info_<run>.log
# vitac_policy_server.py rebuilds the entire deployed model from those two JSON
# sidecars rather than hardcoded constants, so all three files must come from the
# SAME run directory -- mixing a checkpoint with another run's config silently
# deploys a mismatched architecture or wrong units.
#
# Two build args: CKPT_DIR selects the run directory, CKPT_FILE selects which
# epoch inside it (the only thing that varies within a run). Both are copied to
# fixed paths so nothing downstream needs editing per run:
#   docker build --build-arg CKPT_DIR=<run-dir> --build-arg CKPT_FILE=<file>.ckpt .
# scripts/contract_local.sh auto-detects both from checkpoints/.
#
# Backbone weights (assets/backbones/*.pth, vendored under vitacformer/) are
# resolved automatically from whichever backbone training_configs.json names.
ARG CKPT_DIR
ARG CKPT_FILE
COPY checkpoints/${CKPT_DIR}/${CKPT_FILE}              /app/checkpoints/policy_best.ckpt
COPY checkpoints/${CKPT_DIR}/training_configs.json     /app/checkpoints/training_configs.json
COPY checkpoints/${CKPT_DIR}/normalizer_config.json    /app/checkpoints/normalizer_config.json

# Set environment variables
ENV VITAC_CKPT_PATH=/app/checkpoints/policy_best.ckpt \
    ORIGAMI_ACTION_HORIZON=25 \
    PYTHONPATH=/app:/app/vitacformer:/app/vitacformer/detr

# Pre-create writable cache directories and set permissions for UID 65532
RUN mkdir -p /tmp/origami-home /tmp/origami-cache/triton /tmp/origami-cache/torch/hub/checkpoints /tmp/origami-ipython /tmp/origami-cache/torchinductor \
    && chmod 0755 /app/entrypoint.sh \
    && chmod -R a+rX /app /tmp/origami-home /tmp/origami-cache /tmp/origami-ipython \
    && chown -R 65532:65532 /app /tmp/origami-home /tmp/origami-cache /tmp/origami-ipython

# Copy licenses folder into /licenses
COPY licenses/ /licenses/
# Grant read/execute permissions to all users 
RUN chmod -R a+rX /licenses

USER 65532:65532

HEALTHCHECK NONE

ENTRYPOINT ["/app/entrypoint.sh"]