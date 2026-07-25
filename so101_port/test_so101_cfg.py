"""App-gated test: spawn the SO101 ArticulationCfg (actuators + home pose +
fix_root_link) and confirm it initializes and settles at the home pose.
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from polaris.environments.so101_robot_cfg import SO101, SO101_HOME_POS


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=1000.0))

    # SO101 prim_path uses {ENV_REGEX_NS}; give it a concrete namespace
    cfg = SO101.replace(prim_path="/World/robot")
    robot = Articulation(cfg)
    sim.reset()

    names = list(robot.data.joint_names)
    print("CFG num_joints =", robot.num_joints, flush=True)
    print("CFG joint_names =", names, flush=True)
    print("CFG actuator_groups =", list(robot.actuators.keys()), flush=True)

    # check the home pose actually got written as the default
    home = torch.tensor([SO101_HOME_POS[n] for n in names], device=robot.device)
    dq = (robot.data.default_joint_pos[0] - home).abs().max().item()
    print("CFG home_pose_applied_maxerr =", float(dq), flush=True)

    # settle under gravity holding the home target; base must stay fixed
    base_z0 = robot.data.root_pos_w[0, 2].item()
    q0 = robot.data.default_joint_pos.clone()
    for _ in range(120):
        robot.set_joint_position_target(q0)
        robot.write_data_to_sim()
        sim.step()
        robot.update(1.0 / 120.0)
    base_z1 = robot.data.root_pos_w[0, 2].item()
    finite = bool(torch.isfinite(robot.data.joint_pos).all().item())
    drift = float((robot.data.joint_pos - q0).abs().max().item())
    print("CFG base_z_drift =", abs(base_z1 - base_z0), flush=True)
    print("CFG q_finite =", finite, "  home_hold_drift =", drift, flush=True)
    print("SO101_CFG_OK =", finite and dq < 1e-4 and abs(base_z1 - base_z0) < 1e-3, flush=True)


main()
simulation_app.close()
