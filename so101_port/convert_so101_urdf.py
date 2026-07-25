"""Convert the SO-101 URDF to a USD articulation via Isaac Lab's UrdfConverter.

Run inside the `polaris` conda env with polaris_env_conda.sh sourced:
    python so101_port/convert_so101_urdf.py
Produces: PolaRiS-Hub/so101/so101.usd
"""

from isaaclab.app import AppLauncher

# launch Isaac Sim headless before importing any omni/isaaclab.sim modules
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os
from pathlib import Path

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

REPO = Path(__file__).resolve().parent.parent
URDF = REPO / "so101_port" / "urdf" / "so101.urdf"
OUT_DIR = REPO / "PolaRiS-Hub" / "so101"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cfg = UrdfConverterCfg(
    asset_path=str(URDF),
    usd_dir=str(OUT_DIR),
    usd_file_name="so101.usd",
    force_usd_conversion=True,
    make_instanceable=False,
    # table-mounted arm: base is fixed to the world
    fix_base=True,
    merge_fixed_joints=True,
    convert_mimic_joints_to_normal_joints=False,
    self_collision=False,
    # inertials/masses come from the URDF (all 8 links have them)
    link_density=0.0,
    # position-controlled joints; runtime gains are overridden by the
    # ImplicitActuatorCfg in so101_robot_cfg.py, these are just USD defaults
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=17.8, damping=0.6),
    ),
)

converter = UrdfConverter(cfg)
print("SO101_USD_PATH=", converter.usd_path)
print("SO101_CONVERT_OK exists=", os.path.exists(converter.usd_path))

simulation_app.close()
