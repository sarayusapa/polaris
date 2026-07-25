"""Open the Isaac Sim GUI and spawn the SO-101 so it can be watched live.
Gently animates the joints in a sine wave so the arm visibly moves.

Run in the `polaris` conda env with polaris_env_conda.sh sourced:
    python so101_port/view_so101_live.py
Close the Isaac Sim window (or Ctrl-C the process) to stop.
"""

from isaaclab.app import AppLauncher

# GUI mode so a viewport window opens on the desktop (DISPLAY :1)
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import math
from pathlib import Path
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

REPO = Path(__file__).resolve().parent.parent
USD = REPO / "PolaRiS-Hub" / "so101" / "so101.usd"

EXPECTED = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0")
    )
    # camera framing the arm
    sim.set_camera_view([0.5, 0.5, 0.35], [0.0, 0.0, 0.1])

    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/ground", cfg_ground)
    cfg_light = sim_utils.DomeLightCfg(intensity=1200.0)
    cfg_light.func("/World/light", cfg_light)

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(USD)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=17.8, damping=0.6)},
    )
    robot = Articulation(robot_cfg)

    sim.reset()

    names = list(robot.data.joint_names)
    print("SO101 num_joints =", robot.num_joints, flush=True)
    print("SO101 joint_names =", names, flush=True)
    print("SO101 missing_expected =", [j for j in EXPECTED if j not in names], flush=True)
    print("SO101 bodies =", list(robot.data.body_names), flush=True)
    print("SO101_LIVE_READY (watch the Isaac Sim window)", flush=True)

    q0 = robot.data.default_joint_pos.clone()
    t = 0.0
    dt = 1.0 / 60.0
    # amplitude per joint (radians); keep gentle
    amp = torch.zeros_like(q0)
    n = min(robot.num_joints, len(EXPECTED))
    for i in range(n):
        amp[0, i] = 0.4
    while simulation_app.is_running():
        target = q0.clone()
        for i in range(n):
            target[0, i] = q0[0, i] + amp[0, i] * math.sin(2.0 * math.pi * 0.15 * t + i)
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        t += dt

    simulation_app.close()


main()
