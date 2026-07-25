"""Verify SO-101 rubric plumbing (robot-side parts, no scene objects needed):
  * ee_frame resolves and its world pos sits at the gripper (reach checker core)
  * SO-101 gripper open/closed detection flips correctly with open_is_large=True
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv

from polaris.environments.so101_cfg import EnvCfg
from polaris.environments.rubrics.so101_rubrics import SO101_GRIPPER_OPEN_THRESHOLD


def gripper_is_open(env, thr):
    robot = env.scene["robot"]
    gpos = robot.data.joint_pos[0][robot.data.joint_names.index("gripper")]
    return bool(gpos >= thr)  # open_is_large=True


def main():
    env = ManagerBasedRLEnv(cfg=EnvCfg())
    env.reset()

    # --- ee_frame (reach checker core) ---
    ee_pos = env.scene["ee_frame"].data.target_pos_w[0].detach().cpu().numpy()
    robot = env.scene["robot"]
    bidx = list(robot.data.body_names).index("gripper_frame_link")
    grip_pos = robot.data.body_pos_w[0, bidx].detach().cpu().numpy()
    ee_finite = bool(np.isfinite(ee_pos).all())
    ee_dist = float(np.linalg.norm(ee_pos[0] - grip_pos))  # target_pos_w may be (T,3)
    print("RUB ee_pos =", np.round(ee_pos, 3).tolist(), flush=True)
    print("RUB ee_finite =", ee_finite, "  ee_dist_to_gripper =", round(ee_dist, 4), flush=True)

    # --- gripper open detection: command OPEN (large) then CLOSED (small) ---
    open_q = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.5]], device=env.device)
    for _ in range(40):
        env.step(open_q)
    open_detected = gripper_is_open(env, SO101_GRIPPER_OPEN_THRESHOLD)
    print("RUB gripper_after_open_cmd_is_open =", open_detected, flush=True)

    close_q = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, -0.1]], device=env.device)
    for _ in range(40):
        env.step(close_q)
    close_detected = gripper_is_open(env, SO101_GRIPPER_OPEN_THRESHOLD)
    print("RUB gripper_after_close_cmd_is_open =", close_detected, flush=True)

    ok = ee_finite and ee_dist < 0.05 and open_detected and (not close_detected)
    print("SO101_RUBRIC_OK =", ok, flush=True)
    env.close()


main()
simulation_app.close()
