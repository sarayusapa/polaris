"""Consolidated GPU-free smoke test for the generic embodiment framework.

Run in the `polaris` conda env (no Isaac Sim launch needed):
    python so101_port/test_embodiment_cpu.py
"""
import numpy as np
import torch

FAILS = []
def check(cond, msg):
    print(("  [pass] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


print("== 1. spec load / validate / round-trip ==")
from polaris.embodiment.spec import EmbodimentSpec
s = EmbodimentSpec.from_yaml("specs/so101.yaml")
check(s.validate() == [], "so101 spec validates")
check(EmbodimentSpec.from_dict(s.to_dict()).to_dict() == s.to_dict(), "spec round-trips")

print("== 2. URDF ingest (general) ==")
from polaris.embodiment.urdf_ingest import spec_from_urdf
so = spec_from_urdf("so101_port/urdf/so101.urdf", "so101")
check(so.joint_names == ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"],
      "SO-101 motor order")
fr = spec_from_urdf(
    "/home/sra/miniconda3/envs/polaris/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/controllers/config/data/lula_franka_gen.urdf",
    "franka")
check(len(fr.joints) == 9 and fr.gripper.joint == "panda_finger_joint1", "Franka ingest (9 joints, finger gripper)")

print("== 3. policy adapter (auto I/O) ==")
from polaris.embodiment.policy_adapter import policy_spec_from_pretrained
ps = policy_spec_from_pretrained("mot-prog/so101_pick_up_wrist_pan_act")
check(ps.state_dim == 6 and ps.action_dim == 6 and ps.chunk_size == 100, "ACT dims/chunk")
check([i.name for i in ps.images] == ["wrist", "shoulder_pan"], "ACT image keys")
cm = ps.camera_mapping(["wrist", "external"])
check(cm["observation.images.wrist"] == "wrist" and cm["observation.images.shoulder_pan"] == "external",
      "camera mapping wrist->wrist, shoulder_pan->external")

print("== 4. spec parity vs hand-written SO-101 ==")
check(all(j.kp == 17.8 and j.kd == 0.6 for j in s.joints), "gains 17.8/0.6")
check(s.joint("gripper").home == 1.5, "gripper home open")
check(s.gripper.open_is_large is True and s.gripper.open_threshold == 0.9, "gripper open direction")
check(s.ee_frame.link == "gripper_frame_link", "ee link")
check(s.cameras["wrist"].parent_link == "gripper_frame_link"
      and s.cameras["wrist"].offset_rot == (0.5, -0.5, 0.5, -0.5), "wrist cam mount")

print("== 5. validate_embodiment harness ==")
from polaris.embodiment.validate import validate_embodiment
rep = validate_embodiment(s, repo_root=".")
print(str(rep))
check(rep.ok, "validate_embodiment(so101) OK")
check(len(rep.deferred_gpu) == 4, "4 GPU checks correctly deferred")

print("== 6. generic LeRobotClient obs-building (no server) ==")
from polaris.policy.lerobot_client import LeRobotClient
c = object.__new__(LeRobotClient)          # bypass __init__ (which would connect)
c.image_keys = ["observation.images.wrist", "observation.images.shoulder_pan"]
c.image_hw = {"observation.images.wrist": [480, 640], "observation.images.shoulder_pan": [480, 640]}
c._cam_map = None
fake_obs = {
    "splat": {"external_cam": np.zeros((720, 1280, 3), np.uint8),
              "wrist_cam": np.zeros((720, 1280, 3), np.uint8)},
    "policy": {"joint_pos": torch.zeros(1, 6)},
}
req = c._build_request(fake_obs, "pick up the object")
check(set(req.keys()) == {"observation/state", "prompt",
                          "observation.images.wrist", "observation.images.shoulder_pan"},
      "request has state + prompt + both image keys")
check(req["observation.images.wrist"].shape == (480, 640, 3), "wrist image resized to 480x640")
check(c._cam_map["observation.images.wrist"] == "wrist_cam", "wrist key -> wrist_cam")
check(c._cam_map["observation.images.shoulder_pan"] == "external_cam", "shoulder_pan key -> external_cam")

print("== 7. module imports ==")
import importlib
for m in ["polaris.embodiment", "polaris.embodiment.builders",
          "polaris.embodiment.tasks", "polaris.embodiment.validate"]:
    importlib.import_module(m)
    print(f"  [pass] import {m}")

print("\nRESULT:", "ALL PASS" if not FAILS else f"FAILS -> {FAILS}")
