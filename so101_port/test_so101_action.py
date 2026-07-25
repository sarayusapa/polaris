"""Verify the SO-101 6-dim continuous joint-position action:
build the env, command a distinct target per joint, and confirm each action
index drives the matching joint (motor order) to its target.
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

from polaris.environments.so101_cfg import EnvCfg, SO101_JOINT_ORDER


def main():
    cfg = EnvCfg()
    env = ManagerBasedRLEnv(cfg=cfg)
    obs, _ = env.reset()

    # action space dim
    adim = env.action_manager.total_action_dim
    print("ACT total_action_dim =", adim, flush=True)
    print("ACT expected =", len(SO101_JOINT_ORDER), flush=True)
    print("ACT term_order =", env.action_manager.active_terms, flush=True)

    # distinct, in-limits target per joint (radians); gripper toward open
    targets = torch.tensor(
        [[0.30, -0.20, 0.25, -0.15, 0.40, 1.00]], device=env.device
    )
    for _ in range(60):
        env.step(targets)

    robot = env.scene["robot"]
    name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}
    reached = {}
    for k, jname in enumerate(SO101_JOINT_ORDER):
        j = name_to_idx[jname]
        reached[jname] = round(float(robot.data.joint_pos[0, j].item()), 3)
    print("ACT commanded =", {n: round(float(targets[0, k]), 3) for k, n in enumerate(SO101_JOINT_ORDER)}, flush=True)
    print("ACT reached   =", reached, flush=True)

    # each joint should be near its commanded target (servo gains are soft, so
    # allow a generous tolerance; the point is index->joint mapping is correct)
    errs = {n: abs(reached[n] - float(targets[0, k])) for k, n in enumerate(SO101_JOINT_ORDER)}
    print("ACT abs_err =", {n: round(v, 3) for n, v in errs.items()}, flush=True)
    max_err = max(errs.values())
    print("ACT max_err =", round(max_err, 3), flush=True)
    print("SO101_ACTION_OK =", (adim == 6 and max_err < 0.25), flush=True)

    env.close()


main()
simulation_app.close()
