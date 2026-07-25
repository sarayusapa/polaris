"""Rubric builders for SO-101 pick-and-place tasks.

The SO-101 gripper servo is OPEN at large joint values and CLOSED at small
ones (opposite of the DROID Robotiq ``finger_joint``), so is_within_xy must be
called with ``gripper_joint="gripper", open_is_large=True``.

``SO101_GRIPPER_OPEN_THRESHOLD`` is the joint value above which the gripper is
considered "released". The SO-101 gripper range is [-0.174, 1.745] rad; ~0.9
is comfortably open. Tune to the real gripper's released angle if needed.
"""

from polaris.environments.rubrics import checkers
from polaris.environments.rubrics.base import Rubric

SO101_GRIPPER_OPEN_THRESHOLD = 0.9


def pick_place_rubric(
    obj_name: str,
    target_name: str,
    reach_threshold: float = 0.15,
    lift_threshold: float = 0.04,
    lift_default_height: float | None = None,
    place_percent: float = 0.8,
    gripper_open_threshold: float = SO101_GRIPPER_OPEN_THRESHOLD,
) -> Rubric:
    """Standard 3-stage pick-place rubric for the SO-101.

    Stages (monotonic progress 0 -> 1):
      1. reach  : end-effector within reach_threshold of the object
      2. lift   : object lifted lift_threshold above its start height (needs #1)
      3. place  : object's xy overlaps target by place_percent AND the gripper
                  is open/released (needs #2)
    """
    return Rubric(
        criteria=[
            checkers.reach(obj_name, threshold=reach_threshold),
            (
                checkers.lift(
                    obj_name,
                    threshold=lift_threshold,
                    default_height=lift_default_height,
                ),
                [0],
            ),
            (
                checkers.is_within_xy(
                    obj_name,
                    target_name,
                    percent_threshold=place_percent,
                    gripper_joint="gripper",
                    open_is_large=True,
                    open_finger_threshold=gripper_open_threshold,
                ),
                [1],
            ),
        ]
    )
