"""validate_embodiment(spec): run all the checks that don't need Isaac Sim, and
report which GPU checks remain. Generalizes the per-step validation we ran by
hand for the SO-101 (spawn/DOF/action/camera/rubric) into one entry point.

Static (GPU-free) checks run here. The dynamic checks (spawn, DOF count, action
index->joint mapping, camera render) require a running app and are listed as
'deferred' with the script that runs them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from polaris.embodiment.spec import EmbodimentSpec


@dataclass
class ValidationReport:
    ok: bool
    passed: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    deferred_gpu: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"validate_embodiment: {'OK' if self.ok else 'PROBLEMS'}"]
        for p in self.passed:
            lines.append(f"  [pass] {p}")
        for p in self.problems:
            lines.append(f"  [FAIL] {p}")
        for p in self.deferred_gpu:
            lines.append(f"  [gpu ] {p}")
        return "\n".join(lines)


def validate_embodiment(spec: EmbodimentSpec, repo_root: str = ".") -> ValidationReport:
    passed, problems = [], []

    def check(cond, ok_msg, fail_msg):
        (passed if cond else problems).append(ok_msg if cond else fail_msg)

    # 1) internal consistency (delegates to spec.validate)
    spec_problems = spec.validate()
    check(not spec_problems, "spec internally consistent", f"spec.validate: {spec_problems}")

    # 2) joints
    check(len(spec.joints) >= 2, f"{len(spec.joints)} joints", "too few joints")
    check(
        all(j.lower is not None and j.upper is not None for j in spec.joints),
        "all joints have limits",
        "some joints missing limits",
    )
    check(
        all(j.lower <= j.home <= j.upper for j in spec.joints
            if j.lower is not None and j.upper is not None),
        "home pose within limits",
        "home pose outside joint limits: "
        + str([j.name for j in spec.joints
               if j.lower is not None and not (j.lower <= j.home <= j.upper)]),
    )

    # 3) gripper
    if spec.gripper:
        check(spec.gripper.joint in spec.joint_names, "gripper joint exists",
              f"gripper joint '{spec.gripper.joint}' not a joint")
        gj = spec.joint(spec.gripper.joint) if spec.gripper.joint in spec.joint_names else None
        if gj and gj.lower is not None:
            check(gj.lower <= spec.gripper.open_value <= gj.upper
                  and gj.lower <= spec.gripper.closed_value <= gj.upper,
                  "gripper open/closed within limits",
                  "gripper open/closed values outside limits")
    else:
        problems.append("no gripper spec")

    # 4) action
    order = spec.action.order or spec.joint_names
    check(set(order) == set(spec.joint_names), "action order covers all joints",
          "action order mismatch")

    # 5) cameras + ee
    check(any(not c.from_scene for c in spec.cameras.values()) or spec.cameras,
          f"{len(spec.cameras)} cameras", "no cameras")
    check(spec.ee_frame is not None, "ee_frame set", "no ee_frame")

    # 6) assets on disk
    if spec.usd_path:
        p = os.path.join(repo_root, spec.usd_path)
        check(os.path.exists(p), f"usd exists ({spec.usd_path})",
              f"usd missing: {spec.usd_path} (run convert_urdf)")
    if spec.urdf_path:
        p = os.path.join(repo_root, spec.urdf_path)
        check(os.path.exists(p), f"urdf exists ({spec.urdf_path})",
              f"urdf missing: {spec.urdf_path} (run fetch_urdf)")

    deferred = [
        "spawn articulation + DOF count == len(joints)",
        "action index->joint mapping",
        "wrist camera renders + mounted on ee link",
        "ee_frame resolves at the ee link",
    ]
    return ValidationReport(ok=not problems, passed=passed, problems=problems,
                            deferred_gpu=deferred)
