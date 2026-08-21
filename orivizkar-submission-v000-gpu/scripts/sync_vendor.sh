#!/usr/bin/env bash
# =============================================================================
# Refresh the vendored model source from vitacformer++.
#
# The image needs policy.py / detr/ / my_utils/ to rebuild the model at
# inference, but cannot depend on the training repo (lerobot, the dataset,
# accelerate). Docker COPY cannot reach outside the build context, so the
# subset is copied into ./vitacformer/ here, BEFORE the build.
#
# Without this the two copies drift silently: training gets the change, the
# container keeps the old code, and you only find out when load_state_dict
# fails on a shape mismatch -- or worse, when it doesn't fail and the deployed
# model quietly differs from the trained one.
#
#   ./scripts/sync_vendor.sh            # sync, then report what changed
#   ./scripts/sync_vendor.sh --check    # exit 1 if stale; changes nothing
#
# scripts/contract_local.sh runs --check before every build.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_DIR="$(dirname "$SCRIPT_DIR")"
SRC="${VITACFORMER_SRC:-$(dirname "$SUBMISSION_DIR")/vitacformer++}"
DST="${SUBMISSION_DIR}/vitacformer"

# Exactly what the inference path imports, traced from vitac_policy_server.py:
#   policy.py -> detr.main -> detr/models/*
#   my_utils/{normalizer,ori_logging}.py
#   policy.py -> train_utils._stats, train_eval_utils.JOINT_GROUPS
# configs.py is pulled in transitively by the detr arg parser.
# Deliberately NOT synced: assets/ (382MB of checkpoints and backbone weights --
# the Dockerfile handles those separately), dataset/ (lerobot-dependent, unused
# at inference), and the training scripts.
FILES=(
  policy.py
  configs.py
  train_utils.py
  train_eval_utils.py
)
DIRS=(
  detr
  my_utils
)
# never carry these across, whatever directory they live in
EXCLUDES=(--exclude='__pycache__' --exclude='*.pyc' --exclude='*.log'
          --exclude='*.time_log' --exclude='*.ckpt' --exclude='*.pth'
          --exclude='.venv*' --exclude='*.ipynb')

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

[ -d "$SRC" ] || { echo "[fail] source not found: $SRC" >&2; exit 1; }
[ -d "$DST" ] || { echo "[fail] vendored dir not found: $DST" >&2; exit 1; }

stale=0
report () { printf '  %-28s %s\n' "$1" "$2"; }

for f in "${FILES[@]}"; do
    [ -f "$SRC/$f" ] || { echo "[fail] missing in source: $f" >&2; exit 1; }
    if ! diff -q "$SRC/$f" "$DST/$f" >/dev/null 2>&1; then
        stale=1
        n=$(diff "$SRC/$f" "$DST/$f" 2>/dev/null | grep -c '^[<>]' || true)
        report "$f" "STALE (${n} changed lines)"
        [ "$CHECK_ONLY" = "1" ] || cp "$SRC/$f" "$DST/$f"
    else
        report "$f" "in sync"
    fi
done

for d in "${DIRS[@]}"; do
    [ -d "$SRC/$d" ] || { echo "[fail] missing in source: $d/" >&2; exit 1; }
    # -n --out-format catches additions and modifications without writing
    changed=$(rsync -rinc --delete "${EXCLUDES[@]}" "$SRC/$d/" "$DST/$d/" | wc -l)
    if [ "$changed" -gt 0 ]; then
        stale=1
        report "$d/" "STALE (${changed} files differ)"
        [ "$CHECK_ONLY" = "1" ] || rsync -rac --delete "${EXCLUDES[@]}" "$SRC/$d/" "$DST/$d/"
    else
        report "$d/" "in sync"
    fi
done

if [ "$CHECK_ONLY" = "1" ]; then
    if [ "$stale" = "1" ]; then
        echo
        echo "[fail] vendored source is STALE -- the image would ship code that differs"
        echo "       from what you trained with. Run: ./scripts/sync_vendor.sh"
        exit 1
    fi
    echo "[ ok ] vendored source matches $SRC"
else
    [ "$stale" = "1" ] && echo && echo "[ ok ] synced from $SRC" || echo "[ ok ] already up to date"
fi
