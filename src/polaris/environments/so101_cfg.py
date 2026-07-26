"""SO-101 environment config for PolaRiS (analog of droid_cfg.py).

Key differences from the DROID/Franka config:
  * robot is the 6-DOF SO-101 (5 arm servos + 1 gripper servo), see
    so101_robot_cfg.py
  * the action space is 6-dim **continuous joint position** in LeRobot motor
    order [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
    gripper] — NOT a 7+binary-gripper command like DROID. This matches how
    LeRobot ACT / pi0 policies drive the SO-101 (all six motors as positions).
    The InferenceClient (so101 client) is responsible for mapping the trained
    policy's output space (normalisation / units) into these raw joint radians.
  * cameras are re-anchored to SO-101 links (see SceneCfg).

The joint targets fed here are absolute positions in radians
(use_default_offset=False, preserve_order=True so the action vector order is
exactly the motor order above).
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.utils import configclass, noise
from pxr import Usd, UsdGeom, UsdPhysics

# reuse the DROID camera-pose patch (fixes a broken IsaacLab 2.3 camera update)
from polaris.environments.droid_cfg import FixedCamera
from polaris.environments.so101_robot_cfg import (
    SO101,
    SO101_ARM_JOINTS,
    SO101_GRIPPER_JOINT,
)

# full motor order (5 arm + gripper) — the action / joint-state convention
SO101_JOINT_ORDER = SO101_ARM_JOINTS + [SO101_GRIPPER_JOINT]


def _lookat_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """wxyz quaternion for a camera at `eye` looking at `target`, OpenGL
    convention (camera -Z = forward, +Y = up)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    z = -fwd  # OpenGL camera looks down -Z
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])  # camera-to-world rotation
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, qx, qy, qz = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, qx, qy, qz = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, qx, qy, qz = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, qx, qy, qz = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return (float(w), float(qx), float(qy), float(qz))


### SceneCfg ###
@configclass
class SceneCfg(InteractiveSceneCfg):
    """SO-101 tabletop scene."""

    robot = SO101

    # wrist camera anchored to the SO-101 gripper frame link (task #6 refines
    # the exact offset/intrinsics to match the real mounted camera)
    wrist_cam = CameraCfg(
        class_type=FixedCamera,
        prim_path="{ENV_REGEX_NS}/robot/gripper_frame_link/wrist_cam",
        height=720,
        width=1280,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="opengl",
        ),
    )

    sphere_light = AssetBaseCfg(
        prim_path="/World/biglight",
        spawn=sim_utils.DomeLightCfg(intensity=1000),
    )

    def __post_init__(self):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # end-effector frame at the SO-101 gripper frame link
        self.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/robot/gripper_frame_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )

    def dynamic_setup(self, environment_path, robot_splat=True, nightmare="", **kwargs):
        """Load the reconstructed scene USD + its cameras/objects (task #9)."""
        environment_path_ = Path(environment_path)
        environment_path = str(environment_path_.resolve())

        scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn=sim_utils.UsdFileCfg(
                usd_path=environment_path,
                activate_contact_sensors=False,
            ),
        )
        self.scene = scene
        # The SO-101 USD (converted from URDF) has no Gaussian-splat data, so it
        # must be ray-traced and composited into the splat views. The splat env
        # composites pixels whose semantic id >= 2 (get_robot_from_sim), so the
        # robot needs the "raytraced" class tag or it is dropped from every
        # camera. Tag it unconditionally (unlike the DROID robot, which ships
        # its own robot splats).
        self.robot.spawn.semantic_tags = [("class", "raytraced")]

        stage = Usd.Stage.Open(environment_path)
        scene_prim = stage.GetPrimAtPath("/World")
        for child in scene_prim.GetChildren():
            name = child.GetName()
            if child.IsA(UsdGeom.Camera):
                pos = child.GetAttribute("xformOp:translate").Get()
                rot = child.GetAttribute("xformOp:orient").Get()
                rot = (
                    rot.GetReal(),
                    rot.GetImaginary()[0],
                    rot.GetImaginary()[1],
                    rot.GetImaginary()[2],
                )
                asset = CameraCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                    height=720,
                    width=1280,
                    data_types=["rgb", "semantic_segmentation"],
                    colorize_semantic_segmentation=False,
                    spawn=None,
                    offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
                )
                setattr(self, name, asset)
            elif UsdPhysics.RigidBodyAPI(child):
                pos = child.GetAttribute("xformOp:translate").Get()
                rot = child.GetAttribute("xformOp:orient").Get()
                rot = (
                    rot.GetReal(),
                    rot.GetImaginary()[0],
                    rot.GetImaginary()[1],
                    rot.GetImaginary()[2],
                )
                asset = RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                    spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                )
                setattr(self, name, asset)

        # Always override the third-person camera to frame the SO-101 at the
        # origin. The reused scene's external_cam is aimed at the Franka's
        # ~0.5 m workspace, beyond the SO-101's ~0.35 m reach, so the smaller
        # arm falls outside that frame. Spawn a fresh camera looking at the
        # arm's mid-workspace (keep the attribute name `external_cam` so the
        # splat obs key is unchanged).
        _eye = (0.5, -0.4, 0.4)
        _target = (0.15, 0.0, 0.15)
        self.external_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/scene/external_cam_so101",
            height=720,
            width=1280,
            data_types=["rgb", "semantic_segmentation"],
            colorize_semantic_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.0476,
                horizontal_aperture=2.5452,
                vertical_aperture=1.4721,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=_eye,
                rot=_lookat_quat(_eye, _target),
                convention="opengl",
            ),
        )


### ActionCfg ###
@configclass
class ActionCfg:
    """6-dim continuous joint-position action in SO-101 motor order.

    The action vector is [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
    wrist_roll, gripper] as absolute joint targets in radians. preserve_order
    keeps the action index order equal to the list order above so it lines up
    with the policy's output; use_default_offset=False means targets are
    absolute (not deltas from the home pose).
    """

    joints = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=SO101_JOINT_ORDER,
        preserve_order=True,
        use_default_offset=False,
    )


### ObsCfg ###
def joint_pos_ordered(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """All six joint positions in SO-101 motor order (matches the action)."""
    robot = env.scene[asset_cfg.name]
    name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}
    idx = [name_to_idx[n] for n in SO101_JOINT_ORDER]
    return robot.data.joint_pos[:, idx]


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=joint_pos_ordered)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class CommandsCfg:
    pass


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    pass


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=7.0)

    observations = ObservationCfg()
    actions = ActionCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.episode_length_s = 30

        self.viewer.eye = (1.5, 0.0, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 4 * 2
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 4 * 2

        self.rerender_on_reset = True

    def dynamic_setup(self, *args):
        self.scene.dynamic_setup(*args)
