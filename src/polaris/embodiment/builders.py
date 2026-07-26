"""Generic IsaacLab config builders driven by an EmbodimentSpec.

These reproduce what so101_robot_cfg.py / so101_cfg.py build by hand, but from a
spec — so a new robot needs only a spec (+ converted USD), no hand-written cfg.

isaaclab is imported lazily inside the functions: the module can be imported
GPU-free, and the builders only touch isaaclab when actually called inside a
running Isaac Sim app (same app-gating as the hand-written cfgs). The full
equivalence check (spec-built SO-101 env == hand-written SO-101 env) runs under
Isaac Sim; see so101_port/test_spec_parity_gpu.py.
"""
from __future__ import annotations

from polaris.embodiment.spec import EmbodimentSpec


def build_articulation_cfg(spec: EmbodimentSpec, prim_path: str = "{ENV_REGEX_NS}/robot"):
    """EmbodimentSpec -> IsaacLab ArticulationCfg (analog of NVIDIA_DROID/SO101)."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    if spec.usd_path is None:
        raise ValueError(f"spec '{spec.name}' has no usd_path (convert the URDF first)")

    home = {j.name: j.home for j in spec.joints}

    # one implicit actuator per joint, grouped by identical gains to stay tidy
    actuators = {
        j.name: ImplicitActuatorCfg(
            joint_names_expr=[j.name],
            effort_limit=j.effort,
            velocity_limit=j.velocity,
            stiffness=j.kp,
            damping=j.kd,
        )
        for j in spec.joints
    }

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=spec.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
                fix_root_link=spec.fixed_base,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos=home
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators=actuators,
    )


def build_action_term(spec: EmbodimentSpec):
    """EmbodimentSpec -> a JointPositionActionCfg over the spec's motor order."""
    import isaaclab.envs.mdp as mdp

    order = spec.action.order or spec.joint_names
    return mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(order),
        preserve_order=True,
        use_default_offset=False,
    )


def build_wrist_camera_cfg(spec: EmbodimentSpec, name: str = "wrist"):
    """EmbodimentSpec camera -> a CameraCfg mounted on a robot link (if any)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from polaris.environments.droid_cfg import FixedCamera

    cam = spec.cameras.get(name)
    if cam is None or cam.from_scene:
        return None
    return CameraCfg(
        class_type=FixedCamera,
        prim_path="{ENV_REGEX_NS}/robot/" + f"{cam.parent_link}/{name}_cam",
        height=cam.height,
        width=cam.width,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cam.focal_length,
            horizontal_aperture=cam.horizontal_aperture,
            vertical_aperture=cam.vertical_aperture,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=cam.offset_pos, rot=cam.offset_rot, convention="opengl"
        ),
    )


def build_ee_frame_cfg(spec: EmbodimentSpec):
    """EmbodimentSpec.ee_frame -> a FrameTransformerCfg (root -> ee link)."""
    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
        FrameTransformerCfg,
        OffsetCfg,
    )

    if spec.ee_frame is None:
        raise ValueError(f"spec '{spec.name}' has no ee_frame")

    marker = FRAME_MARKER_CFG.copy()
    marker.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker.prim_path = "/Visuals/FrameTransformer"
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/base_link",
        debug_vis=False,
        visualizer_cfg=marker,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/" + spec.ee_frame.link,
                name="end_effector",
                offset=OffsetCfg(pos=list(spec.ee_frame.offset)),
            )
        ],
    )


def joint_pos_observation(spec: EmbodimentSpec):
    """Return an obs function giving joint positions in the spec's motor order."""
    order = spec.action.order or spec.joint_names

    def _obs(env, asset_cfg=None):
        robot = env.scene["robot"]
        name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}
        idx = [name_to_idx[n] for n in order]
        return robot.data.joint_pos[:, idx]

    return _obs


def _lookat_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """wxyz quaternion for a camera at eye looking at target (opengl convention)."""
    import numpy as np

    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    z = -fwd; x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    R = np.column_stack([x, y, z]); t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        q = (0.25 * s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s)
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2]) * 2
        q = ((R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s)
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1 + R[1,1] - R[0,0] - R[2,2]) * 2
        q = ((R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s)
    else:
        s = np.sqrt(1 + R[2,2] - R[0,0] - R[1,1]) * 2
        q = ((R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s)
    return tuple(float(v) for v in q)


def build_env_cfg(spec: EmbodimentSpec, external_eye=(0.5, -0.4, 0.4),
                  external_target=(0.15, 0.0, 0.15)):
    """EmbodimentSpec -> a full ManagerBasedRLEnvCfg class (the generic analog of
    so101_cfg.EnvCfg). Returns a configclass suitable for `env_cfg_entry_point`.

    Reuses build_articulation_cfg / build_action_term / build_wrist_camera_cfg /
    build_ee_frame_cfg / joint_pos_observation. dynamic_setup loads the scene
    USD, frames the robot with a repositioned external cam, and tags the robot
    `raytraced` so a URDF-converted (splat-less) robot composites into the splat
    views.
    """
    from pathlib import Path

    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass
    from pxr import Usd, UsdGeom, UsdPhysics

    robot_cfg = build_articulation_cfg(spec)
    wrist_cam_cfg = build_wrist_camera_cfg(spec)
    action_term = build_action_term(spec)
    obs_func = joint_pos_observation(spec)

    def _dynamic_setup(scene, environment_path, robot_splat=True, **kwargs):
        environment_path = str(Path(environment_path).resolve())
        scene.scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn=sim_utils.UsdFileCfg(usd_path=environment_path, activate_contact_sensors=False),
        )
        # URDF-imported robots carry no splat; tag raytraced so they composite
        scene.robot.spawn.semantic_tags = [("class", "raytraced")]

        stage = Usd.Stage.Open(environment_path)
        for child in stage.GetPrimAtPath("/World").GetChildren():
            name = child.GetName()
            if child.IsA(UsdGeom.Camera):
                pos = child.GetAttribute("xformOp:translate").Get()
                rot = child.GetAttribute("xformOp:orient").Get()
                rot = (rot.GetReal(), *rot.GetImaginary())
                setattr(scene, name, CameraCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/scene/{name}", height=720, width=1280,
                    data_types=["rgb", "semantic_segmentation"],
                    colorize_semantic_segmentation=False, spawn=None,
                    offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl")))
            elif UsdPhysics.RigidBodyAPI(child):
                pos = child.GetAttribute("xformOp:translate").Get()
                rot = child.GetAttribute("xformOp:orient").Get()
                rot = (rot.GetReal(), *rot.GetImaginary())
                setattr(scene, name, RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/scene/{name}", spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot)))

        # always frame the robot (scene cams target the original embodiment)
        scene.external_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/scene/external_cam_gen", height=720, width=1280,
            data_types=["rgb", "semantic_segmentation"], colorize_semantic_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(focal_length=1.0476, horizontal_aperture=2.5452,
                                             vertical_aperture=1.4721),
            offset=CameraCfg.OffsetCfg(pos=external_eye,
                                       rot=_lookat_quat(external_eye, external_target),
                                       convention="opengl"))

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = robot_cfg
        wrist_cam = wrist_cam_cfg
        sphere_light = AssetBaseCfg(prim_path="/World/biglight",
                                    spawn=sim_utils.DomeLightCfg(intensity=1000))

        def __post_init__(self):
            self.ee_frame = build_ee_frame_cfg(spec)

        def dynamic_setup(self, environment_path, robot_splat=True, nightmare="", **kwargs):
            _dynamic_setup(self, environment_path, robot_splat, **kwargs)

    @configclass
    class ObservationCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos = ObsTerm(func=obs_func)

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

        policy: ObsGroup = PolicyCfg()

    @configclass
    class ActionCfg:
        joints = action_term

    @configclass
    class EventCfg:
        reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    @configclass
    class TerminationsCfg:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

    @configclass
    class RewardsCfg:
        pass

    @configclass
    class CommandsCfg:
        pass

    @configclass
    class CurriculumCfg:
        pass

    @configclass
    class EnvCfg(ManagerBasedRLEnvCfg):
        scene = SceneCfg(num_envs=1, env_spacing=7.0)
        observations = ObservationCfg()
        actions = ActionCfg()
        rewards = RewardsCfg()
        commands = CommandsCfg()
        curriculum = CurriculumCfg()
        terminations = TerminationsCfg()
        events = EventCfg()

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

    return EnvCfg
