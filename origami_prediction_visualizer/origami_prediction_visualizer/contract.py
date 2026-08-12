"""North 65-joint contract used by the prediction visualizer."""

from __future__ import annotations

ACTION_DIM = 65

def _hand_joint_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_thumb_CMC_FE", f"{side}_thumb_CMC_AA",
        f"{side}_thumb_MCP_FE", f"{side}_thumb_MCP_AA",
        f"{side}_thumb_IP",
        f"{side}_index_MCP_FE", f"{side}_index_MCP_AA",
        f"{side}_index_PIP", f"{side}_index_DIP",
        f"{side}_middle_MCP_FE", f"{side}_middle_MCP_AA",
        f"{side}_middle_PIP", f"{side}_middle_DIP",
        f"{side}_ring_MCP_FE", f"{side}_ring_MCP_AA",
        f"{side}_ring_PIP", f"{side}_ring_DIP",
        f"{side}_pinky_CMC", f"{side}_pinky_MCP_FE",
        f"{side}_pinky_MCP_AA", f"{side}_pinky_PIP",
        f"{side}_pinky_DIP",
    )

JOINT_NAMES = (
    tuple(f"left_arm_joint_{i}" for i in range(1, 8))
    + _hand_joint_names("left")
    + tuple(f"right_arm_joint_{i}" for i in range(1, 8))
    + _hand_joint_names("right")
    + tuple(f"lower_body_joint_{i}" for i in range(1, 6))
    + ("neck_joint_1", "neck_joint_2")
)

JOINT_GROUPS = (
    ("left_arm", 0, 7),
    ("left_hand", 7, 29),
    ("right_arm", 29, 36),
    ("right_hand", 36, 58),
    ("motor", 58, 65),
)

assert len(JOINT_NAMES) == ACTION_DIM
