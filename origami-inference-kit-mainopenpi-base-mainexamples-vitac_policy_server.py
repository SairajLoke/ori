#!/usr/bin/env python3
"""ViTacFormer policy server.

Run with:
    python viTac_policy_server.py --port 8000
Then check the contract:
    python sharpa_north_ces_lite_sdk-main/examples/check_policy_server.py \
        --policy-host 127.0.0.1 --policy-port 8000
"""

from __future__ import annotations
import argparse
import logging
import numpy as np
import torch

# OpenPI server utilities
from openpi_client import base_policy as _base_policy
from openpi.serving import websocket_policy_server

# Import the ViTacFormer ACTPolicy and the training config
from policy import ACTPolicy  # viTacFormer/policy.py
from configs import (
    CAMERA_NAMES,
    TACTILE_TEMPORAL_HORIZON,
    TACTILE_TEMPORAL_TOTAL_TIMESTAMPS,
    PROPRIOCEPTIVE_TEMPORAL_HORIZON,
    CHUNK_SIZE,
    STATE_DIM,
    LR_BACKNNE,
    BACKBONE,
    # any other hyper‑params you used during training – copy them below
)

# ---------------------------------------------------------------------------
# Helper: build the exact config dict that was used for training.
# You can copy the dict from `origami_imitate_episodes.py` or from a saved JSON.
# ---------------------------------------------------------------------------
def build_policy_config() -> dict:
    return {
        "lr": 1e-4,                     # example – replace with your value
        "num_queries": CHUNK_SIZE,
        "kl_weight": 1.0,               # example
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "lr_backbone": 1e-5,
        "backbone": BACKBONE,
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": CAMERA_NAMES,
        "use_tactile": True,
        "state_dim": STATE_DIM,
        "proprioceptive_temporal_horizon": PROPRIOCEPTIVE_TEMPORAL_HORIZON,
    }

# ---------------------------------------------------------------------------
# Wrapper that enforces the tactile contract and calls ACTPolicy.
# ---------------------------------------------------------------------------
class ViTacPolicy(_base_policy.BasePolicy):
    def __init__(self, config: dict):
        self.policy = ACTPolicy(config)
        self.policy.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    def _check_and_convert(self, obs: dict):
        # ---- low‑dim tactile vector (19, 60) ----
        if "observation/tactile" not in obs:
            raise KeyError("Missing required key: observation/tactile")
        tactile = np.asarray(obs["observation/tactile"], dtype=np.float32)
        if tactile.shape != (TACTILE_TEMPORAL_HORIZON * 2 + 1, 60):
            raise ValueError(
                f"observation/tactile must have shape {(TACTILE_TEMPORAL_HORIZON * 2 + 1, 60)}; got {tactile.shape}"
            )
        # ---- deform grid history (19, 480, 1200, 3) ----
        if "observation/image/tactile_deform" not in obs:
            raise KeyError("Missing required key: observation/image/tactile_deform")
        deform = np.asarray(obs["observation/image/tactile_deform"], dtype=np.uint8)
        if deform.shape != (TACTILE_TEMPORAL_HORIZON * 2 + 1, 480, 1200, 3):
            raise ValueError(
                f"observation/image/tactile_deform must have shape {(TACTILE_TEMPORAL_HORIZON * 2 + 1, 480, 1200, 3)}; got {deform.shape}"
            )
        # ---- joint state (65,) ----
        if "observation/state" not in obs:
            raise KeyError("Missing required key: observation/state")
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"observation/state must have shape {(STATE_DIM,)}; got {state.shape}")
        # ---- camera images (4 cams, H, W, 3) ----
        cams = []
        for name in ["head_left", "head_right", "wrist_left", "wrist_right"]:
            key = f"observation/image/{name}"
            if key not in obs:
                raise KeyError(f"Missing required camera image: {key}")
            img = np.asarray(obs[key], dtype=np.uint8)
            cams.append(img)
        # Convert everything to torch tensors on the correct device
        qpos = torch.from_numpy(state).unsqueeze(0).to(self.device)  # (1, 65)
        image = torch.stack(
            [torch.from_numpy(c).permute(2, 0, 1) for c in cams], dim=0
        ).unsqueeze(0).float().to(self.device)  # (1, 4, 3, H, W)
        tactile_t = torch.from_numpy(tactile).unsqueeze(0).to(self.device)  # (1, 19, 60)
        deform_t = torch.from_numpy(deform).unsqueeze(0).to(self.device)  # (1, 19, 480, 1200, 3)
        return qpos, image, tactile_t, deform_t

    def infer(self, obs: dict) -> dict:
        qpos, image, tactile, deform = self._check_and_convert(obs)
        # epoch=999 disables teacher‑forcing (inference mode)
        a_hat = self.policy(
            qpos, image, None, None, self.device, tactile, deform, epoch=999
        )
        actions = a_hat.squeeze(0).cpu().numpy().astype(np.float32)
        return {"actions": actions}

    def reset(self) -> None:
        # No per‑episode state for this model
        pass

# ---------------------------------------------------------------------------
# Main entry point – identical to the template server.
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    policy_config = build_policy_config()
    policy = ViTacPolicy(policy_config)
    metadata = {
        "policy": "ViTacFormer",
        "action_dim": 65,
        "action_horizon": 25,  # keep the same horizon as the template
    }
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    logging.info("Serving ViTacFormer policy on %s:%d", args.host, args.port)
    server.serve_forever()

if __name__ == "__main__":
    main()
