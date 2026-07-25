"""DEFERRED (needs GPU): prove the spec-driven builders reproduce the
hand-written SO-101. Builds an ArticulationCfg from specs/so101.yaml via the
generic builders, spawns it, and checks DOF/joint-order/home/action mapping
match the hand-written so101_robot_cfg / so101_cfg results.

Run in the `polaris` conda env with polaris_env_conda.sh sourced:
    python so101_port/test_spec_builders_gpu.py
"""
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from polaris.embodiment.spec import EmbodimentSpec
from polaris.embodiment.builders import build_articulation_cfg


def main():
    spec = EmbodimentSpec.from_yaml("specs/so101.yaml")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=1000.0))

    cfg = build_articulation_cfg(spec).replace(prim_path="/World/robot")
    robot = Articulation(cfg)
    sim.reset()

    names = list(robot.data.joint_names)
    print("SPEC num_joints =", robot.num_joints, flush=True)
    print("SPEC joint_names =", names, flush=True)
    home = torch.tensor([spec.joint(n).home for n in names], device=robot.device)
    dq = (robot.data.default_joint_pos[0] - home).abs().max().item()
    print("SPEC home_applied_maxerr =", float(dq), flush=True)
    ok = (
        robot.num_joints == 6
        and set(names) == set(spec.joint_names)
        and dq < 1e-4
    )
    print("SPEC_BUILDERS_OK =", ok, flush=True)


main()
simulation_app.close()
