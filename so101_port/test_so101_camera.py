"""Verify the SO-101 wrist camera: prim path resolves, it renders a valid RGB
image, and its world pose sits at the gripper frame (i.e. it's actually mounted
on the arm). Saves a PNG of the wrist view.
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from pathlib import Path
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

from polaris.environments.so101_cfg import EnvCfg

OUT = Path("/tmp/claude-1000/-home-sra-sarayu-polaris/633619cc-0dec-4da2-af36-6ddc4cccb8eb/scratchpad/so101_wrist_view.png")


def main():
    cfg = EnvCfg()
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()

    # put the arm in a pose that brings the gripper over the workspace so the
    # wrist cam looks at something other than empty sky
    q = torch.tensor([[0.0, -0.6, 0.8, -0.6, 0.0, 1.0]], device=env.device)
    for _ in range(60):
        env.step(q)

    cam = env.scene["wrist_cam"]
    rgb = cam.data.output["rgb"]  # (N, H, W, C)
    print("CAM rgb_shape =", tuple(rgb.shape), flush=True)
    arr = rgb[0].detach().cpu().numpy()
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    print("CAM dtype =", arr.dtype, " min =", int(arr.min()), " max =", int(arr.max()), flush=True)
    nonempty = int(arr.max()) > 0
    print("CAM nonempty =", nonempty, flush=True)

    # camera world pos vs gripper_frame_link world pos -> should coincide
    cam_pos = cam.data.pos_w[0].detach().cpu().numpy()
    robot = env.scene["robot"]
    bidx = list(robot.data.body_names).index("gripper_frame_link")
    grip_pos = robot.data.body_pos_w[0, bidx].detach().cpu().numpy()
    dist = float(np.linalg.norm(cam_pos - grip_pos))
    print("CAM cam_pos =", np.round(cam_pos, 3).tolist(), flush=True)
    print("CAM gripper_pos =", np.round(grip_pos, 3).tolist(), flush=True)
    print("CAM mount_dist_to_gripper =", round(dist, 4), flush=True)

    # save PNG
    try:
        import imageio.v2 as imageio
        imageio.imwrite(OUT, arr.astype(np.uint8))
        print("CAM saved =", str(OUT), flush=True)
    except Exception as e:
        print("CAM save_failed =", repr(e), flush=True)

    ok = (rgb.shape[0] == 1 and rgb.shape[-1] in (3, 4) and nonempty and dist < 0.05)
    print("SO101_CAMERA_OK =", ok, flush=True)
    env.close()


main()
simulation_app.close()
