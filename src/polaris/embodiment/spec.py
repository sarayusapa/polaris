"""Declarative embodiment specification.

One EmbodimentSpec captures everything robot-specific that we otherwise
hand-write per arm (see so101_robot_cfg.py / so101_cfg.py as the reference
implementation this generalizes). Generic builders (builders.py) turn a spec
into IsaacLab configs; the URDF ingest (urdf_ingest.py) auto-drafts a spec from
a URDF; the policy adapter (policy_adapter.py) auto-derives a policy's I/O.

This module is pure Python (no isaaclab / no GPU) so it can be authored, loaded,
validated, and unit-tested without launching Isaac Sim.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

Units = Literal["radians", "degrees", "normalized"]


@dataclass
class JointSpec:
    name: str
    lower: float | None = None          # rad (from URDF)
    upper: float | None = None
    home: float = 0.0                    # rad
    kp: float = 17.8                     # implicit-actuator stiffness
    kd: float = 0.6                      # damping
    effort: float = 10.0
    velocity: float = 10.0


@dataclass
class GripperSpec:
    joint: str                          # which joint is the gripper
    open_is_large: bool = True          # SO-101: open at large angle; DROID: small
    open_value: float = 1.5             # rad target for "open"
    closed_value: float = -0.1          # rad target for "closed"
    # threshold (rad) above/below which the rubric treats the gripper as open
    open_threshold: float = 0.9


@dataclass
class CameraSpec:
    parent_link: str | None = None      # robot link it mounts to (None = from scene)
    offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
    width: int = 1280
    height: int = 720
    focal_length: float = 2.8
    horizontal_aperture: float = 5.376
    vertical_aperture: float = 3.024
    from_scene: bool = False            # True = provided by the scene USD

    def __post_init__(self):
        # YAML round-trips tuples as lists; normalize so equality/consumers are stable
        self.offset_pos = tuple(self.offset_pos)
        self.offset_rot = tuple(self.offset_rot)


@dataclass
class EEFrameSpec:
    link: str
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        self.offset = tuple(self.offset)


@dataclass
class ActionSpec:
    type: Literal["joint_position"] = "joint_position"
    units: Units = "radians"            # units the POLICY emits (server converts)
    order: list[str] = field(default_factory=list)  # joint order of the action vector


@dataclass
class EmbodimentSpec:
    name: str
    usd_path: str | None = None
    urdf_path: str | None = None
    fixed_base: bool = True
    joints: list[JointSpec] = field(default_factory=list)   # ordered = motor order
    gripper: GripperSpec | None = None
    action: ActionSpec = field(default_factory=ActionSpec)
    state_units: Units = "radians"      # units the policy expects for joint state
    cameras: dict[str, CameraSpec] = field(default_factory=dict)
    ee_frame: EEFrameSpec | None = None

    # -- convenience --
    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.joints]

    @property
    def arm_joint_names(self) -> list[str]:
        g = self.gripper.joint if self.gripper else None
        return [j.name for j in self.joints if j.name != g]

    def joint(self, name: str) -> JointSpec:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(name)

    # -- (de)serialization --
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EmbodimentSpec":
        d = dict(d)
        d["joints"] = [JointSpec(**j) for j in d.get("joints", [])]
        if d.get("gripper"):
            d["gripper"] = GripperSpec(**d["gripper"])
        if d.get("action"):
            d["action"] = ActionSpec(**d["action"])
        if d.get("ee_frame"):
            d["ee_frame"] = EEFrameSpec(**d["ee_frame"])
        d["cameras"] = {k: CameraSpec(**v) for k, v in d.get("cameras", {}).items()}
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "EmbodimentSpec":
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def validate(self) -> list[str]:
        """Return a list of problems (empty = valid)."""
        problems = []
        if not self.joints:
            problems.append("no joints")
        names = self.joint_names
        if len(names) != len(set(names)):
            problems.append("duplicate joint names")
        if self.gripper and self.gripper.joint not in names:
            problems.append(f"gripper joint '{self.gripper.joint}' not in joints")
        if self.action.order and set(self.action.order) != set(names):
            problems.append("action.order does not match joint set")
        if self.ee_frame is None:
            problems.append("no ee_frame (needed by reach/lift rubric)")
        if self.usd_path is None and self.urdf_path is None:
            problems.append("neither usd_path nor urdf_path set")
        return problems
