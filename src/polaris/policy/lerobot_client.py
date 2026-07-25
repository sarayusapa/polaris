"""Generic PolaRiS client for any LeRobot policy served by so101_port/serve_lerobot.py.

Unlike the hand-written so101_client.py (fixed to wrist+shoulder_pan), this
client self-configures from the server's metadata: the server sends the policy's
image keys + resolutions (derived from the checkpoint config via
policy_adapter), and the client maps each to one of the embodiment's rendered
cameras (obs["splat"]). One client serves ACT, SmolVLA, pi0, ... unchanged.

Unit conversion (deg/rad/normalized) lives in the server (it owns the policy).
"""
import numpy as np
from openpi_client import websocket_client_policy, image_tools

from polaris.policy.abstract_client import InferenceClient, PolicyArgs


@InferenceClient.register(client_name="LeRobot")
class LeRobotClient(InferenceClient):
    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        if args.open_loop_horizon is None:
            raise ValueError("open_loop_horizon must be set for LeRobotClient")
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        meta = self.client.get_server_metadata() if hasattr(
            self.client, "get_server_metadata"
        ) else {}
        # metadata contract from serve_lerobot.py
        self.image_keys: list[str] = meta.get("image_keys", [])
        self.image_hw: dict = meta.get("image_hw", {})
        self.open_loop_horizon = args.open_loop_horizon
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self._cam_map = None  # policy image key -> embodiment cam name (lazy)

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def _map_cameras(self, splat_cams: list[str]) -> dict:
        """policy image key -> embodiment camera name (wrist matches wrist, the
        rest fall through to remaining cameras in order)."""
        mapping, used = {}, set()
        for key in self.image_keys:
            short = key.split(".")[-1]
            match = None
            for cam in splat_cams:
                if cam in short or short in cam:
                    match = cam
                    break
            if match is None:
                remaining = [c for c in splat_cams if c not in used]
                match = remaining[0] if remaining else (splat_cams[0] if splat_cams else None)
            mapping[key] = match
            used.add(match)
        return mapping

    def _build_request(self, obs, instruction):
        splat = obs["splat"]
        if self._cam_map is None:
            self._cam_map = self._map_cameras(list(splat.keys()))
        state = obs["policy"]["joint_pos"].clone().detach().cpu().numpy()[0].astype(np.float32)
        req = {"observation/state": state, "prompt": instruction}
        for key in self.image_keys:
            cam = self._cam_map[key]
            hw = self.image_hw.get(key, [480, 640])
            req[key] = image_tools.resize_with_pad(splat[cam], int(hw[0]), int(hw[1]))
        return req

    def _viz(self, obs):
        splat = obs["splat"]
        cams = list(splat.keys())
        tiles = [image_tools.resize_with_pad(splat[c], 224, 224) for c in cams[:2]]
        return np.concatenate(tiles, axis=1) if tiles else None

    def infer(self, obs, instruction, return_viz: bool = False):
        both = None
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0
            resp = self.client.infer(self._build_request(obs, instruction))
            self.pred_action_chunk = np.asarray(resp["actions"])
            both = self._viz(obs)

        if return_viz and both is None:
            both = self._viz(obs)
        if self.pred_action_chunk is None:
            raise ValueError("No action chunk predicted")

        action = np.asarray(self.pred_action_chunk[self.actions_from_chunk_completed])
        self.actions_from_chunk_completed += 1
        return action, both
