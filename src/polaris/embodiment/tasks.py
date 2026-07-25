"""Generic task/rubric templates parameterized by an EmbodimentSpec.

Generalizes so101_rubrics.pick_place_rubric: instead of hard-coding the gripper
joint + open direction, they are read from the spec, so the same template works
for any robot. `checkers` is imported lazily (it pulls omni.usd, so it is only
importable inside a running Isaac Sim app) — this module stays importable
GPU-free.
"""
from __future__ import annotations

from polaris.embodiment.spec import EmbodimentSpec


def pick_place_rubric(
    spec: EmbodimentSpec,
    obj_name: str,
    target_name: str,
    reach_threshold: float = 0.15,
    lift_threshold: float = 0.04,
    lift_default_height: float | None = None,
    place_percent: float = 0.8,
):
    """3-stage pick-place rubric (reach -> lift -> place) using the spec's
    gripper convention for the 'released' check in the place stage."""
    from polaris.environments.rubrics import checkers
    from polaris.environments.rubrics.base import Rubric

    if spec.gripper is None:
        raise ValueError(f"spec '{spec.name}' has no gripper; cannot build pick_place")
    g = spec.gripper

    return Rubric(
        criteria=[
            checkers.reach(obj_name, threshold=reach_threshold),
            (
                checkers.lift(obj_name, threshold=lift_threshold,
                              default_height=lift_default_height),
                [0],
            ),
            (
                checkers.is_within_xy(
                    obj_name, target_name, percent_threshold=place_percent,
                    gripper_joint=g.joint, open_is_large=g.open_is_large,
                    open_finger_threshold=g.open_threshold,
                ),
                [1],
            ),
        ]
    )
