"""Standalone smoke test (run in the `lerobot` conda env): load the SO-101 ACT
checkpoint + its processors, run one dummy inference, print the action chunk
shape and value range (range reveals whether actions are radians or degrees).
"""
import numpy as np
import torch

REPO = "mot-prog/so101_pick_up_wrist_pan_act"
DEVICE = "cuda"

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

policy = ACTPolicy.from_pretrained(REPO)
policy.eval().to(DEVICE)
cfg = policy.config
print("config input_features:", list(cfg.input_features.keys()))
print("config output_features:", list(cfg.output_features.keys()))

pre, post = make_pre_post_processors(cfg, pretrained_path=REPO)

# dummy observation matching the config: state[6], two 3x480x640 images
batch = {
    "observation.state": torch.zeros(1, 6, dtype=torch.float32),
    "observation.images.wrist": torch.rand(1, 3, 480, 640, dtype=torch.float32),
    "observation.images.shoulder_pan": torch.rand(1, 3, 480, 640, dtype=torch.float32),
    "task": ["pick up the object"],
}

proc = pre(batch)
print("after preprocess keys:", list(proc.keys()))
with torch.no_grad():
    chunk = policy.predict_action_chunk(proc)
print("raw chunk type:", type(chunk), "shape:", tuple(chunk.shape))
out = post(chunk)
out = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
print("post-processed action shape:", out.shape)
print("action min/max/mean:", float(out.min()), float(out.max()), float(out.mean()))
print("first action vector:", np.round(out.reshape(-1, out.shape[-1])[0], 3).tolist())
print("SMOKE_LEROBOT_OK")
