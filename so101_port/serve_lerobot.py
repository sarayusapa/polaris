"""Generic LeRobot policy server over the openpi websocket protocol.

Generalizes serve_lerobot_act.py to any LeRobot policy: it derives the I/O
contract from the checkpoint config (via policy_adapter), dispatches to the
right policy class by config type, and advertises its image keys + resolutions
to the client (LeRobotClient) as connection metadata. Pairs with
src/polaris/policy/lerobot_client.py.

Run in the `lerobot` conda env:
    conda run -n lerobot python so101_port/serve_lerobot.py \
        --repo <hf_repo_or_path> --port 8000 \
        --state-units degrees --action-units degrees

Units: the sim is radians; declare the units the policy was trained in so the
server converts state (rad->policy) in and action (policy->rad) out. Supported:
'radians' (identity) and 'degrees' (x180/pi and back).
"""
import argparse
import asyncio
import traceback

import numpy as np
import torch
from openpi_client import msgpack_numpy
import websockets
import websockets.asyncio.server as _server

# reuse the policy adapter to read the checkpoint's I/O contract
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from polaris.embodiment.policy_adapter import policy_spec_from_pretrained  # noqa: E402

_UNIT = {"radians": 1.0, "degrees": np.pi / 180.0}  # policy-units -> radians


def _load_policy(repo: str, device: str):
    """Dispatch to the right LeRobot policy class by checkpoint type."""
    from lerobot.policies.factory import make_pre_post_processors

    ps = policy_spec_from_pretrained(repo)
    t = ps.policy_type
    if t == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy as P
    elif t == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as P
    elif t == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as P
    elif t == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as P
    else:
        raise ValueError(f"unsupported policy type '{t}'")
    policy = P.from_pretrained(repo).eval().to(device)
    # checkpoints bake the training device into the processor pipeline; force it
    # to this server's device so the same checkpoint runs on CPU or any GPU.
    dev_override = {"device_processor": {"device": device}}
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=repo,
        preprocessor_overrides=dev_override,
        postprocessor_overrides=dev_override,
    )
    return policy, pre, post, ps


class LeRobotServer:
    def __init__(self, repo, device="cuda", state_units="degrees", action_units="degrees"):
        self.device = device
        self.policy, self.pre, self.post, self.ps = _load_policy(repo, device)
        self.rad_to_policy_state = 1.0 / _UNIT[state_units]   # radians -> policy units
        self.policy_to_rad_action = _UNIT[action_units]       # policy units -> radians
        print(f"[serve] {repo} type={self.ps.policy_type} "
              f"state={self.ps.state_dim} action={self.ps.action_dim} "
              f"imgs={self.ps.image_names} on {device}", flush=True)

    def metadata(self) -> dict:
        return {
            "policy_type": self.ps.policy_type,
            "image_keys": [im.key for im in self.ps.images],
            "image_hw": {im.key: [im.height, im.width] for im in self.ps.images},
            "state_dim": self.ps.state_dim,
            "action_dim": self.ps.action_dim,
        }

    def _img(self, arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device)
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        return t.permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        state_rad = np.asarray(obs["observation/state"], dtype=np.float32).reshape(-1)
        state = torch.from_numpy(state_rad * self.rad_to_policy_state).float().unsqueeze(0).to(self.device)
        batch = {"observation.state": state, "task": [obs.get("prompt", "")]}
        for im in self.ps.images:
            batch[im.key] = self._img(obs[im.key])
        proc = self.pre(batch)
        chunk = self.policy.predict_action_chunk(proc)
        chunk = self.post(chunk).detach().cpu().numpy()[0].astype(np.float32)
        return {"actions": chunk * self.policy_to_rad_action}


async def _run(server: LeRobotServer, host, port):
    packer = msgpack_numpy.Packer()
    meta = server.metadata()

    async def handler(ws):
        await ws.send(packer.pack(meta))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await ws.recv())
                await ws.send(packer.pack(server.infer(obs)))
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception:
                await ws.send(traceback.format_exc())
                raise

    async with _server.serve(handler, host, port, compression=None, max_size=None):
        print(f"[serve] listening on {host}:{port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--state-units", default="degrees", choices=list(_UNIT))
    ap.add_argument("--action-units", default="degrees", choices=list(_UNIT))
    a = ap.parse_args()
    srv = LeRobotServer(a.repo, a.device, a.state_units, a.action_units)
    asyncio.run(_run(srv, a.host, a.port))
