"""Serve a LeRobot SO-101 ACT policy over the openpi websocket protocol.

Run in the `lerobot` conda env:
    conda run -n lerobot python so101_port/serve_lerobot_act.py \
        --repo mot-prog/so101_pick_up_wrist_pan_act --port 8000

Protocol (matches openpi_client.WebsocketClientPolicy, used by SO101Client):
  request obs dict:
    observation/state            : (6,) float, sim joint positions in RADIANS
    observation/wrist_image       : (H,W,3) uint8
    observation/shoulder_pan_image: (H,W,3) uint8
    prompt                        : str (ignored by ACT)
  response dict:
    actions : (chunk, 6) float, joint targets in RADIANS

Unit conversion lives here (the server owns the policy): the ACT policy is
trained in DEGREES, the sim is in RADIANS, so state is deg=rad*180/pi on the way
in and action is rad=deg*pi/180 on the way out (uniform over all 6 joints; the
gripper's 0..100 deg maps onto the URDF's 0..1.745 rad open range).
"""
import argparse
import asyncio
import traceback

import numpy as np
import torch
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server

RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0


class ACTPolicyServer:
    def __init__(self, repo: str, device: str = "cuda"):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.device = device
        self.policy = ACTPolicy.from_pretrained(repo).eval().to(device)
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=repo
        )
        print(f"[serve] loaded {repo} on {device}", flush=True)

    def _img(self, arr):
        # HWC uint8 -> 1xCxHxW float [0,1] on device
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device)
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        return t.permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        state_rad = np.asarray(obs["observation/state"], dtype=np.float32).reshape(-1)
        state_deg = torch.from_numpy(state_rad * RAD2DEG).float().unsqueeze(0).to(self.device)
        batch = {
            "observation.state": state_deg,
            "observation.images.wrist": self._img(obs["observation/wrist_image"]),
            "observation.images.shoulder_pan": self._img(obs["observation/shoulder_pan_image"]),
            "task": [obs.get("prompt", "")],
        }
        proc = self.pre(batch)
        chunk = self.policy.predict_action_chunk(proc)  # (1, T, 6) degrees
        chunk = self.post(chunk)
        chunk = chunk.detach().cpu().numpy()[0].astype(np.float32)  # (T, 6) degrees
        return {"actions": chunk * DEG2RAD}  # -> radians


async def _run(server: ACTPolicyServer, host: str, port: int):
    packer = msgpack_numpy.Packer()

    async def handler(ws):
        await ws.send(packer.pack({"policy": "so101_act"}))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await ws.recv())
                action = server.infer(obs)
                await ws.send(packer.pack(action))
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception:
                await ws.send(traceback.format_exc())
                raise

    import websockets
    async with _server.serve(handler, host, port, compression=None, max_size=None):
        print(f"[serve] listening on {host}:{port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    import websockets  # noqa: F401  (ensure available)

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="mot-prog/so101_pick_up_wrist_pan_act")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    srv = ACTPolicyServer(args.repo, args.device)
    asyncio.run(_run(srv, args.host, args.port))
