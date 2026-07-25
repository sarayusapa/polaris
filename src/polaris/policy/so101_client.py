import numpy as np
from openpi_client import websocket_client_policy, image_tools

from polaris.policy.abstract_client import InferenceClient, PolicyArgs

# SO-101 wrist camera is trained at 480x640 (see the ACT checkpoint config).
_IMG_H, _IMG_W = 480, 640


@InferenceClient.register(client_name="SO101")
class SO101Client(InferenceClient):
    """Client for a LeRobot SO-101 policy served over the openpi websocket
    protocol (see so101_port/serve_lerobot_act.py).

    Sends the 6-dim joint state (sim radians) + wrist and shoulder_pan camera
    images; receives a chunk of 6-dim joint-position actions. Unit conversion
    between sim radians and the policy's native action space lives in the
    SERVER (it owns the policy + its normalization stats), so this client is
    embodiment-plumbing only: state out, action in, both 6-dim.
    """

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        if args.open_loop_horizon is None:
            raise ValueError("open_loop_horizon must be set for SO101Client")
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.open_loop_horizon = args.open_loop_horizon

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        both = None
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            curr = self._extract_observation(obs)
            self.actions_from_chunk_completed = 0

            shoulder_img = image_tools.resize_with_pad(
                curr["shoulder_pan_image"], _IMG_H, _IMG_W
            )
            wrist_img = image_tools.resize_with_pad(
                curr["wrist_image"], _IMG_H, _IMG_W
            )
            request_data = {
                "observation/state": curr["state"],
                "observation/wrist_image": wrist_img,
                "observation/shoulder_pan_image": shoulder_img,
                "prompt": instruction,
            }
            server_response = self.client.infer(request_data)
            self.pred_action_chunk = np.asarray(server_response["actions"])
            both = np.concatenate(
                [
                    image_tools.resize_with_pad(curr["shoulder_pan_image"], 224, 224),
                    image_tools.resize_with_pad(curr["wrist_image"], 224, 224),
                ],
                axis=1,
            )

        if return_viz and both is None:
            curr = self._extract_observation(obs)
            both = np.concatenate(
                [
                    image_tools.resize_with_pad(curr["shoulder_pan_image"], 224, 224),
                    image_tools.resize_with_pad(curr["wrist_image"], 224, 224),
                ],
                axis=1,
            )

        if self.pred_action_chunk is None:
            raise ValueError("No action chunk predicted")

        action = np.asarray(self.pred_action_chunk[self.actions_from_chunk_completed])
        self.actions_from_chunk_completed += 1

        # 6-dim continuous joint targets (no gripper binarization for SO-101)
        return action, both

    def _extract_observation(self, obs_dict):
        # rendered camera images (side-channel), external_cam -> shoulder_pan view
        shoulder_pan_image = obs_dict["splat"]["external_cam"]
        wrist_image = obs_dict["splat"]["wrist_cam"]

        # 6-dim joint state in motor order (radians), from the ObservationCfg
        joint_pos = obs_dict["policy"]["joint_pos"].clone().detach().cpu().numpy()[0]

        return {
            "shoulder_pan_image": shoulder_pan_image,
            "wrist_image": wrist_image,
            "state": joint_pos.astype(np.float32),
        }
