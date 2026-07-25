"""SO-101 (SO-ARM101) articulation config for PolaRiS.

Analogous to `NVIDIA_DROID` in robot_cfg.py, but for the LeRobot SO-101 arm:
a 5-DOF arm (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
plus a 1-DOF gripper, all Feetech STS3215 servos.

The USD is produced from the official SO-ARM101 URDF by
`so101_port/convert_so101_urdf.py` (Isaac Lab UrdfConverter).

Joint limits (from so101_new_calib.urdf, radians):
    shoulder_pan   [-1.91986, 1.91986]
    shoulder_lift  [-1.74533, 1.74533]
    elbow_flex     [-1.69000, 1.69000]
    wrist_flex     [-1.65806, 1.65806]
    wrist_roll     [-2.74385, 2.84121]
    gripper        [-0.17453, 1.74533]   (lower = closed, upper = open)
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from polaris.utils import DATA_PATH

# Ordered arm joints (matches the LeRobot Feetech bus motor order 1..5).
SO101_ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
SO101_GRIPPER_JOINT = "gripper"

# A neutral "ready" home pose: arm upright with a slight elbow bend, gripper open.
SO101_HOME_POS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    # gripper near the open end of its range
    "gripper": 1.5,
}

SO101 = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(DATA_PATH / "so101/so101.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
            # the base is a table-mounted fixed root (fix_base=True at conversion)
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=SO101_HOME_POS,
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        # 5 arm servos: position-controlled STS3215. Gains tuned for the small
        # servo scale (URDF effort/velocity limits are 10 Nm / 10 rad/s).
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            effort_limit=10.0,
            velocity_limit=10.0,
            stiffness=17.8,
            damping=0.6,
        ),
        # gripper servo
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit=10.0,
            velocity_limit=10.0,
            stiffness=17.8,
            damping=0.6,
        ),
    },
)
