"""
Centralised logging for the Origami ViTacFormer pipeline.

Goal: trace the flow of data end-to-end (dataset -> convert_batch -> policy ->
backbone/transformer -> loss) with zero cost when switched off, and without
each rank of a multi-GPU run spamming the same lines four times.

--------------------------------------------------------------------------
QUICK START
--------------------------------------------------------------------------
In the entrypoint (origami_imitate_episodes.py), once, after Accelerator init:

    from my_utils.ori_logging import setup_logging
    setup_logging(rank=accelerator.process_index, log_dir=ckpt_dir)

Everywhere else:

    from my_utils.ori_logging import get_logger, TRACE, log_tensor
    log = get_logger("data")          # -> logger named "ori.data"
    log.info("dataset has %d frames", len(ds))
    log_tensor(log, TRACE, "action", action)

--------------------------------------------------------------------------
CONTROLLING IT (all via env vars, no code edits)
--------------------------------------------------------------------------
  ORI_LOG_LEVEL=INFO          global level. TRACE|DEBUG|INFO|WARNING|ERROR|OFF
                              OFF disables the whole "ori" tree.
  ORI_LOG_LEVELS="data=TRACE,model=DEBUG"
                              per-subsystem overrides. Bare names are relative
                              to "ori." so "data" == "ori.data".
  ORI_LOG_RANKS=0             which ranks may log. "0" (default) | "all" | "0,2"
  ORI_LOG_FILE=/path/log      explicit file target (default <log_dir>/ori_debug.log)
  ORI_LOG_CONSOLE=1           1 (default) = also log to stderr, 0 = file only
  ORI_LOG_TENSOR_STATS=1      1 (default) = log_tensor computes min/max/mean/std.
                              0 = shapes/dtype/device only. Set 0 if the .item()
                              syncs perturb your timing measurements.

Subsystem names in use: data, model, train, norm, policy, backbone, transformer

--------------------------------------------------------------------------
LEVEL CONVENTIONS USED IN THIS CODEBASE
--------------------------------------------------------------------------
  INFO   once-per-run facts: config, dataset size, module dims, param counts,
         resume/checkpoint events, per-epoch summaries.
  DEBUG  once-per-step structural facts: batch keys, token layout, loss terms.
         Safe to leave on; no per-element GPU work.
  TRACE  per-tensor statistics (shape/dtype/device/min/max/mean/std/nan).
         Forces GPU syncs -> slows training. Use for a handful of steps.

--------------------------------------------------------------------------
WHY THE isEnabledFor GUARDS MATTER
--------------------------------------------------------------------------
Calling tensor.min().item() on a CUDA tensor forces a device sync. If that ran
unconditionally it would serialise the training loop even when logging is off.
log_tensor()/log_tensors() check the level FIRST and return immediately, so a
disabled TRACE call costs one integer comparison.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# TRACE level (below DEBUG) for per-tensor dumps
# ---------------------------------------------------------------------------
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

ROOT_NAME = "ori"

_CONFIGURED = False
_RANK = 0
_RANK_ENABLED = True
_TENSOR_STATS = True


def _parse_level(name: str, default: int = logging.INFO) -> int:
    """'trace'/'debug'/'off'/'25' -> numeric level."""
    if name is None:
        return default
    name = str(name).strip().upper()
    if name in ("OFF", "NONE", "DISABLED"):
        return logging.CRITICAL + 10  # nothing gets through
    if name == "TRACE":
        return TRACE
    if name.isdigit():
        return int(name)
    return getattr(logging, name, default)


def _rank_may_log(rank: int) -> bool:
    spec = os.environ.get("ORI_LOG_RANKS", "0").strip().lower()
    if spec in ("all", "*"):
        return True
    try:
        return rank in {int(x) for x in spec.split(",") if x.strip() != ""}
    except ValueError:
        return rank == 0


def setup_logging(
    rank: int = 0,
    log_dir: Optional[str] = None,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure the 'ori' logger tree. Idempotent; safe to call more than once.

    Args:
        rank:     this process's global rank. Non-permitted ranks are silenced
                  entirely (see ORI_LOG_RANKS).
        log_dir:  directory for the default log file (usually ckpt_dir).
        level:    overrides ORI_LOG_LEVEL if given.
        log_file: overrides ORI_LOG_FILE if given.
    """
    global _CONFIGURED, _RANK, _RANK_ENABLED, _TENSOR_STATS

    _RANK = rank
    _RANK_ENABLED = _rank_may_log(rank)
    _TENSOR_STATS = os.environ.get("ORI_LOG_TENSOR_STATS", "1") not in ("0", "false", "False")

    root = logging.getLogger(ROOT_NAME)

    # Reconfiguring (e.g. called again once the real rank is known) -> drop old
    # handlers, and clear any per-subsystem levels left over from a previous
    # call. Without the reset a child level set earlier would keep overriding
    # the new root level (including ORI_LOG_LEVEL=OFF).
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    _prefix = ROOT_NAME + "."
    for _name in list(logging.Logger.manager.loggerDict):
        if _name.startswith(_prefix):
            _child = logging.getLogger(_name)
            if isinstance(_child, logging.Logger):
                _child.setLevel(logging.NOTSET)

    root.propagate = False  # don't leak into whatever lerobot/accelerate set up

    if not _RANK_ENABLED:
        root.setLevel(logging.CRITICAL + 10)
        root.addHandler(logging.NullHandler())
        _CONFIGURED = True
        return root

    lvl = _parse_level(level if level is not None else os.environ.get("ORI_LOG_LEVEL", "INFO"))
    root.setLevel(lvl)

    fmt = logging.Formatter(
        fmt=f"%(asctime)s [r{rank}] %(levelname)-5s %(name)-16s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if os.environ.get("ORI_LOG_CONSOLE", "1") not in ("0", "false", "False"):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    path = log_file or os.environ.get("ORI_LOG_FILE")
    if path is None and log_dir is not None:
        path = os.path.join(log_dir, f"ori_debug_rank{rank}.log")
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fh = logging.FileHandler(path, mode="a")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Per-subsystem overrides: ORI_LOG_LEVELS="data=TRACE,model=DEBUG"
    # ORI_LOG_LEVEL=OFF is a hard kill and deliberately outranks these, so that
    # one env var is always enough to silence the whole tree.
    _off = lvl > logging.CRITICAL
    for item in ("" if _off else os.environ.get("ORI_LOG_LEVELS", "")).split(","):
        if "=" not in item:
            continue
        name, _, val = item.partition("=")
        name = name.strip()
        if not name:
            continue
        if not name.startswith(ROOT_NAME + "."):
            name = f"{ROOT_NAME}.{name}"
        logging.getLogger(name).setLevel(_parse_level(val))

    _CONFIGURED = True
    root.debug("logging configured: level=%s file=%s tensor_stats=%s",
               logging.getLevelName(lvl), path, _TENSOR_STATS)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return the 'ori.<name>' logger. Works before setup_logging()."""
    if name.startswith(ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")


# ---------------------------------------------------------------------------
# Tensor introspection
# ---------------------------------------------------------------------------

def tensor_repr(t, stats: Optional[bool] = None) -> str:
    """Compact one-line description of a tensor.

    With stats=True this calls .min()/.max()/.mean()/.std(), each of which forces
    a CUDA sync. Only reachable through the level-guarded helpers below.
    """
    if t is None:
        return "None"
    if not hasattr(t, "shape"):
        return f"<{type(t).__name__}>"

    head = f"shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}"

    use_stats = _TENSOR_STATS if stats is None else stats
    if not use_stats or getattr(t, "numel", lambda: 0)() == 0:
        return head

    try:
        import torch  # local import: this module must be importable without torch
        f = t.detach().float()
        parts = [
            f"min={f.min().item():.4g}",
            f"max={f.max().item():.4g}",
            f"mean={f.mean().item():.4g}",
        ]
        if f.numel() > 1:
            parts.append(f"std={f.std().item():.4g}")
        if torch.is_floating_point(t):
            n_nan = int(torch.isnan(f).sum().item())
            n_inf = int(torch.isinf(f).sum().item())
            if n_nan or n_inf:
                parts.append(f"** NaN={n_nan} Inf={n_inf} **")
        return head + " " + " ".join(parts)
    except Exception as exc:  # never let logging break training
        return head + f" <stats failed: {exc}>"


def log_tensor(logger: logging.Logger, level: int, name: str, t, stats: Optional[bool] = None) -> None:
    """Log one tensor. No-op (and no GPU sync) if `level` is disabled."""
    if not logger.isEnabledFor(level):
        return
    logger.log(level, "%-28s %s", name, tensor_repr(t, stats=stats))


def log_tensors(logger: logging.Logger, level: int, prefix: str, mapping, stats: Optional[bool] = None) -> None:
    """Log a dict / iterable of (name, tensor) pairs under a common prefix."""
    if not logger.isEnabledFor(level):
        return
    items = mapping.items() if hasattr(mapping, "items") else mapping
    for k, v in items:
        if isinstance(k, str) and k.startswith("_"):
            continue  # skip bookkeeping entries like "_timing"
        log_tensor(logger, level, f"{prefix}{k}", v, stats=stats)


def log_module_shapes(logger: logging.Logger, level: int, module, max_params: int = 0) -> None:
    """Log a module's parameter count, and optionally each parameter's shape."""
    if not logger.isEnabledFor(level):
        return
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logger.log(level, "%s: %.2fM params (%.2fM trainable)",
               type(module).__name__, total / 1e6, trainable / 1e6)
    if max_params:
        for i, (n, p) in enumerate(module.named_parameters()):
            if i >= max_params:
                logger.log(level, "  ... (%d more)", total - i)
                break
            logger.log(level, "  %-50s %s", n, tuple(p.shape))


# ---------------------------------------------------------------------------
# Throttling — for anything inside the training loop
# ---------------------------------------------------------------------------

class StepGate:
    """Rate-limit logging inside hot loops.

    Typical use in a forward() that runs thousands of times:

        _gate = StepGate(first_n=3, every=500)
        ...
        if _gate() and log.isEnabledFor(TRACE):
            log_tensor(log, TRACE, "src", src)

    Logs the first `first_n` calls, then one call in every `every`.
    """

    __slots__ = ("first_n", "every", "n")

    def __init__(self, first_n: int = 3, every: int = 0):
        self.first_n = first_n
        self.every = every
        self.n = 0

    def __call__(self) -> bool:
        i = self.n
        self.n += 1
        if i < self.first_n:
            return True
        return bool(self.every) and (i % self.every == 0)

    def reset(self) -> None:
        self.n = 0


def is_rank_logging() -> bool:
    """True if this rank is permitted to emit logs at all."""
    return _RANK_ENABLED


__all__ = [
    "TRACE", "setup_logging", "get_logger", "tensor_repr",
    "log_tensor", "log_tensors", "log_module_shapes", "StepGate", "is_rank_logging",
]
