# ViTacFormer Policy Server for Origami Competition
# Multi-stage Docker build for minimal runtime CPU image

# =============================================================================
# Stage 1: Build dependencies
# =============================================================================
FROM python:3.11-slim AS builder

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN python -m venv /.venv
ENV PATH="/.venv/bin:${PATH}"

# torch/torchvision need the special --index-url, so installed separately from
# the rest. No build-essential needed here -- torch/torchvision/opencv/msgpack
# all ship prebuilt wheels for this platform, nothing gets compiled.
RUN pip install \
    --default-timeout=100 \
    --retries=5 \
    --index-url "${TORCH_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    torch torchvision
# --no-cache-dir     

# Everything else comes from requirements-inference.txt (from earlier in this
# thread) -- referenced here, not retyped, so the two files can't drift out of
# sync. Expected in the build context alongside this Dockerfile. NOTE:
# msgpack-numpy was dropped -- vitac_policy_server.py only imports plain
# msgpack (it uses its own object_hook-based numpy codec, matching the SDK's
# own policy_server_template.py -- msgpack-numpy was never actually used).
COPY requirements-inference.txt /tmp/requirements-inference.txt
RUN pip install \
    --index-url https://pypi.org/simple \
    -r /tmp/requirements-inference.txt \
    && rm /tmp/requirements-inference.txt
#  --no-cache-dir

# ---------------------------------------------------------------------------
# Stage 2: runtime -- minimal, nonroot, no build tools
# ---------------------------------------------------------------------------


FROM python:3.11-slim AS runtime

# Install system libraries required by OpenCV, NumPy, and PyTorch C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /.venv /.venv
ENV PATH="/.venv/bin:${PATH}"

WORKDIR /app

# vitacformer inference-only source (no dataset/, no lerobot dependency).
# Not pip-installed -- resolved purely via the PYTHONPATH env var set below,
# so a plain COPY is all that's needed. (No `pip install -e` step: that was
# redundant with PYTHONPATH and, since the builder stage's copy of detr/
# never made it into this runtime stage anyway, was silently relying on the
# two stages happening to use the same absolute path -- fragile and
# unnecessary.)
COPY vitacformer/policy.py       /app/vitacformer/policy.py
COPY vitacformer/detr            /app/vitacformer/detr
COPY vitacformer/train_utils.py  /app/vitacformer/train_utils.py
COPY vitacformer/configs.py      /app/vitacformer/configs.py

# The single-file protocol adapter (replaces the old policy_server.py +
# vitac_policy/ package).
COPY vitac_policy_server.py /app/vitac_policy_server.py

COPY entrypoint.sh /app/entrypoint.sh

# Checkpoint baked into the image (read-only at runtime). Adjust the source
# filename to match whatever you actually place in checkpoints/.
COPY checkpoints/policy_epoch_30_loss_0.041.ckpt /app/checkpoints/policy_best.ckpt
COPY checkpoints/resnet18-f37072fd.pth /tmp/origami-cache/torch/hub/checkpoints/resnet18-f37072fd.pth

ENV TORCHINDUCTOR_CACHE_DIR=/app/.cache/torchinductor \
    VITAC_CKPT_PATH=/app/checkpoints/policy_best.ckpt \
    ORIGAMI_ACTION_HORIZON=25 \
    PYTHONPATH=/app/vitacformer:/app/vitacformer/detr \
    HOME=/tmp/origami-home \
    XDG_CACHE_HOME=/tmp/origami-cache \
    IPYTHONDIR=/tmp/origami-ipython
# vitac_policy_server.py resolves `policy.py`/`detr`/etc. purely through this
# PYTHONPATH -- it does not touch sys.path itself. Two entries are required,
# not one: /app/vitacformer lets `import detr` / `import detr.main` resolve,
# but detr/models/backbone.py and detr/models/position_encoding.py do their
# own `from util.misc import ...` as an ABSOLUTE import (not package-relative
# to detr), so detr/ itself must also be directly on the path for `import
# util` to resolve to detr/util/misc.py. Omitting the second entry produces
# "No module named 'detr'"/"No module named 'util'" errors deep inside the
# model construction call, not at the top-level import.
# If you ever move where vitacformer/ gets COPYed to above, update both.
# MPLCONFIGDIR was dropped -- matplotlib isn't imported anywhere in the
# inference path (policy.py/detr/train_utils.py).

# Pre-create writable-cache dirs so the image also works when run WITHOUT the
# evaluator's tmpfs mounts (e.g. a plain `docker run`); under the evaluator's
# --read-only + tmpfs /tmp, these paths are shadowed by the tmpfs mount at
# runtime anyway.
RUN mkdir -p /tmp/origami-home /tmp/origami-cache /tmp/origami-ipython \
    && chmod 0755 /app/entrypoint.sh \
    && chown -R 65532:65532 /app /tmp/origami-home /tmp/origami-cache /tmp/origami-ipython

# Matches the --user 65532:65532 the local evaluator forces via `docker run`
# regardless (docker_runtime.py's _policy_command); set here too so the image
# is correct standalone.
USER 65532:65532

HEALTHCHECK NONE

ENTRYPOINT ["/app/entrypoint.sh"]