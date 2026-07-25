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
