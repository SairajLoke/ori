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
# ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
# RUN --mount=type=cache,target=/root/.cache/pip \
#     python3 -m pip install \
#     --default-timeout=100 \
#     --retries=5 \
#     --index-url "${TORCH_INDEX_URL}" \
#     --extra-index-url https://pypi.org/simple \
#     torch torchvision msgpack
#     # --no-cache-dir

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
COPY entrypoint.sh /app/entrypoint.sh

# Copy baked checkpoints and hub assets
COPY checkpoints/policy_epoch_30_loss_0.041.ckpt /app/checkpoints/policy_best.ckpt
COPY checkpoints/resnet18-f37072fd.pth           /app/checkpoints/resnet18-f37072fd.pth

# Set environment variables
ENV VITAC_CKPT_PATH=/app/checkpoints/policy_best.ckpt \
    BACKBONE_PATH=/app/checkpoints/resnet18-f37072fd.pth \
    ORIGAMI_ACTION_HORIZON=25 \
    PYTHONPATH=/app/vitacformer:/app/vitacformer/detr

# Pre-create writable cache directories and set permissions for UID 65532
RUN mkdir -p /tmp/origami-home /tmp/origami-cache/triton /tmp/origami-cache/torch/hub/checkpoints /tmp/origami-ipython /tmp/origami-cache/torchinductor \
    && chmod 0755 /app/entrypoint.sh \
    && chmod -R a+rX /app /tmp/origami-home /tmp/origami-cache /tmp/origami-ipython \
    && chown -R 65532:65532 /app /tmp/origami-home /tmp/origami-cache /tmp/origami-ipython

USER 65532:65532

HEALTHCHECK NONE

ENTRYPOINT ["/app/entrypoint.sh"]