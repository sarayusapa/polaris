"""Generic embodiment framework: declarative robot specs + generic builders.

Generalizes the hand-written SO-101 port (so101_robot_cfg.py / so101_cfg.py /
so101_client.py) so a new robot is "convert URDF + fill a spec + point at a
checkpoint" instead of three hand-authored modules.

Pure-Python entry points (no isaaclab / GPU): spec, urdf_ingest, policy_adapter.
The isaaclab config builders live in builders.py (imported lazily).
"""
from polaris.embodiment.spec import (
    ActionSpec,
    CameraSpec,
    EEFrameSpec,
    EmbodimentSpec,
    GripperSpec,
    JointSpec,
)

__all__ = [
    "EmbodimentSpec",
    "JointSpec",
    "GripperSpec",
    "CameraSpec",
    "EEFrameSpec",
    "ActionSpec",
]
