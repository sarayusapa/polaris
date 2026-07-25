"""GPU-free smoke of the generic LeRobot server: load the ACT policy on CPU,
build the LeRobotServer, and run one inference on dummy obs. Verifies the
metadata contract, the batch build, and the deg->rad action conversion.

Run in the `lerobot` conda env:
    conda run -n lerobot python so101_port/test_serve_lerobot_cpu.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "so101_port"))
import serve_lerobot as S

REPO = "mot-prog/so101_pick_up_wrist_pan_act"

srv = S.LeRobotServer(REPO, device="cpu", state_units="degrees", action_units="degrees")

meta = srv.metadata()
print("metadata:", meta)
assert meta["image_keys"] == ["observation.images.wrist", "observation.images.shoulder_pan"], meta
assert meta["state_dim"] == 6 and meta["action_dim"] == 6

obs = {
    "observation/state": np.zeros(6, np.float32),  # radians
    "observation.images.wrist": np.zeros((480, 640, 3), np.uint8),
    "observation.images.shoulder_pan": np.zeros((480, 640, 3), np.uint8),
    "prompt": "pick up the object",
}
out = srv.infer(obs)
act = np.asarray(out["actions"])
print("action shape:", act.shape, "| finite:", bool(np.isfinite(act).all()),
      "| radians range:", round(float(act.min()), 3), round(float(act.max()), 3))
assert act.shape == (100, 6), act.shape
assert np.isfinite(act).all()
# deg->rad: a policy in degrees (~tens) becomes ~<2 rad after conversion
assert abs(act).max() < 6.5, "actions not in a plausible radian range"
print("SERVE_LEROBOT_CPU_OK")
