"""Spawn the converted SO-101 USD in an empty Isaac Sim scene and validate the
articulation: DOF count, joint names, and that it initializes + holds a home
pose without a PhysX/articulation-view crash.

Run in the `polaris` conda env with polaris_env_conda.sh sourced:
    python so101_port/validate_so101_usd.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pathlib import Path
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

REPO = Path(__file__).resolve().parent.parent
USD = REPO / "PolaRiS-Hub" / "so101" / "so101.usd"

EXPECTED_JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0")
    )
    sim.set_camera_view([0.6, 0.6, 0.4], [0.0, 0.0, 0.1])

    # ground + light
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=1000.0))

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(USD)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=17.8,
                damping=0.6,
            ),
        },
    )
    robot = Articulation(robot_cfg)

    sim.reset()

    # ---- validation ----
    names = list(robot.data.joint_names)
    dof = robot.num_joints
    print("VALIDATE num_joints =", dof)
    print("VALIDATE joint_names =", names)
    missing = [j for j in EXPECTED_JOINTS if j not in names]
    print("VALIDATE missing_expected =", missing)
    print("VALIDATE bodies =", list(robot.data.body_names))

    # step the sim a bit holding the default pose; confirm it stays finite
    q0 = robot.data.default_joint_pos.clone()
    for _ in range(120):
        robot.set_joint_position_target(q0)
        robot.write_data_to_sim()
        sim.step()
        robot.update(1.0 / 120.0)
    q_final = robot.data.joint_pos
    print("VALIDATE q_final_finite =", bool(torch.isfinite(q_final).all().item()))
    print("VALIDATE max_drift_from_home =", float((q_final - q0).abs().max().item()))
    ok = (not missing) and torch.isfinite(q_final).all().item()
    print("SO101_VALIDATE_OK =", ok)

main()
simulation_app.close()
