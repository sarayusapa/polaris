"""Draft an EmbodimentSpec from a URDF (pure XML, no GPU).

Extracts actuated joints (revolute/prismatic) ordered by the kinematic chain
(base -> tip, which is the natural motor order), their limits, the link tree,
and heuristic guesses for the end-effector link and gripper joint. The result
is a *template*: the fields a URDF can't tell you (camera mounts, action units,
gripper open direction, USD path) are left at sensible defaults for a human to
confirm.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from polaris.embodiment.spec import (
    ActionSpec,
    EEFrameSpec,
    EmbodimentSpec,
    GripperSpec,
    JointSpec,
)

_ACTUATED = {"revolute", "continuous", "prismatic"}
_EE_HINTS = ("gripper_frame", "tool", "tcp", "ee", "end_effector", "flange", "hand")
_GRIP_HINTS = ("gripper", "finger", "jaw", "grip")


def _chain_order(joints_raw: list[dict]) -> list[dict]:
    """Order actuated joints base->tip by walking the link tree from the root."""
    children = {}  # parent_link -> list[joint]
    child_links = set()
    for j in joints_raw:
        children.setdefault(j["parent"], []).append(j)
        child_links.add(j["child"])
    all_parents = {j["parent"] for j in joints_raw}
    roots = [p for p in all_parents if p not in child_links]
    root = roots[0] if roots else (joints_raw[0]["parent"] if joints_raw else None)

    ordered, seen = [], set()
    stack = [root]
    while stack:
        link = stack.pop(0)
        for j in children.get(link, []):
            if j["name"] in seen:
                continue
            seen.add(j["name"])
            ordered.append(j)
            stack.append(j["child"])
    # append any actuated joints not reached (disconnected) to be safe
    for j in joints_raw:
        if j["name"] not in seen:
            ordered.append(j)
    return ordered


def _guess_ee_link(links: list[str], joints_raw_all: list[dict]) -> str:
    for hint in _EE_HINTS:
        for ln in links:
            if hint in ln.lower():
                return ln
    # else: the deepest leaf link (a child that is never a parent)
    parents = {j["parent"] for j in joints_raw_all}
    leaves = [j["child"] for j in joints_raw_all if j["child"] not in parents]
    return leaves[-1] if leaves else (links[-1] if links else "base_link")


def _guess_gripper(ordered: list[dict]) -> str | None:
    for j in ordered:
        if any(h in j["name"].lower() for h in _GRIP_HINTS):
            return j["name"]
    return ordered[-1]["name"] if ordered else None


def spec_from_urdf(urdf_path: str, name: str | None = None) -> EmbodimentSpec:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = [l.get("name") for l in root.findall("link")]

    joints_all, actuated = [], []
    for j in root.findall("joint"):
        jd = {
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": (j.find("parent").get("link") if j.find("parent") is not None else None),
            "child": (j.find("child").get("link") if j.find("child") is not None else None),
        }
        lim = j.find("limit")
        if lim is not None:
            jd["lower"] = _f(lim.get("lower"))
            jd["upper"] = _f(lim.get("upper"))
            jd["effort"] = _f(lim.get("effort"))
            jd["velocity"] = _f(lim.get("velocity"))
        joints_all.append(jd)
        if jd["type"] in _ACTUATED:
            actuated.append(jd)

    ordered = _chain_order(actuated)

    joint_specs = [
        JointSpec(
            name=j["name"],
            lower=j.get("lower"),
            upper=j.get("upper"),
            home=0.0,
            effort=j.get("effort") or 10.0,
            velocity=j.get("velocity") or 10.0,
        )
        for j in ordered
    ]

    grip_joint = _guess_gripper(ordered)
    gripper = GripperSpec(joint=grip_joint) if grip_joint else None
    ee_link = _guess_ee_link(links, joints_all)

    return EmbodimentSpec(
        name=name or _stem(urdf_path),
        urdf_path=urdf_path,
        fixed_base=True,
        joints=joint_specs,
        gripper=gripper,
        action=ActionSpec(order=[j.name for j in joint_specs]),
        ee_frame=EEFrameSpec(link=ee_link),
    )


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _stem(path: str) -> str:
    import os
    return os.path.splitext(os.path.basename(path))[0]
