"""Auto-derive a policy's I/O contract from a LeRobot checkpoint (no GPU).

This generalizes what we hand-coded in so101_client.py / serve_lerobot_act.py:
a LeRobot checkpoint's config.json already declares its input_features
(state dim + camera image keys/shapes), output_features (action dim), chunk
size, and normalization. Reading it lets a single generic client+server serve
*any* LeRobot policy instead of one hand-written client per checkpoint.

The only thing config.json can't give us is the units the policy was trained in
(degrees vs radians vs normalized) — that is declared once on the embodiment /
policy spec and applied by the generic server.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ImageInput:
    key: str                      # e.g. "observation.images.wrist"
    name: str                     # short name, e.g. "wrist"
    channels: int
    height: int
    width: int


@dataclass
class PolicySpec:
    policy_type: str              # "act", "smolvla", ...
    state_dim: int
    action_dim: int
    images: list[ImageInput] = field(default_factory=list)
    chunk_size: int = 1
    n_action_steps: int = 1
    normalization: dict | None = None
    repo: str | None = None

    @property
    def image_names(self) -> list[str]:
        return [im.name for im in self.images]

    def camera_mapping(self, embodiment_cameras: list[str]) -> dict[str, str]:
        """Map each policy image input -> an embodiment camera name.

        Heuristic: exact/substring name match (wrist->wrist), else fall back to
        the remaining embodiment cameras in order (e.g. shoulder_pan/front/top
        -> external).
        """
        mapping, used = {}, set()
        for im in self.images:
            match = None
            for cam in embodiment_cameras:
                if cam in im.name or im.name in cam:
                    match = cam
                    break
            if match is None:
                remaining = [c for c in embodiment_cameras if c not in used]
                match = remaining[0] if remaining else (embodiment_cameras[0] if embodiment_cameras else None)
            mapping[im.key] = match
            used.add(match)
        return mapping


def _short_name(key: str) -> str:
    # "observation.images.shoulder_pan" -> "shoulder_pan"
    return key.split(".")[-1]


def policy_spec_from_config(config: dict, repo: str | None = None) -> PolicySpec:
    inf = config.get("input_features", {})
    outf = config.get("output_features", {})

    state_dim = 0
    images: list[ImageInput] = []
    for key, feat in inf.items():
        ftype = feat.get("type")
        shape = feat.get("shape", [])
        if ftype == "STATE":
            state_dim = int(shape[0]) if shape else 0
        elif ftype == "VISUAL":
            c, h, w = (shape + [None, None, None])[:3]
            images.append(ImageInput(key=key, name=_short_name(key), channels=c, height=h, width=w))

    action_dim = 0
    for key, feat in outf.items():
        if feat.get("type") == "ACTION":
            shp = feat.get("shape", [])
            action_dim = int(shp[0]) if shp else 0

    return PolicySpec(
        policy_type=config.get("type", "unknown"),
        state_dim=state_dim,
        action_dim=action_dim,
        images=images,
        chunk_size=int(config.get("chunk_size", 1) or 1),
        n_action_steps=int(config.get("n_action_steps", 1) or 1),
        normalization=config.get("normalization_mapping"),
        repo=repo,
    )


def policy_spec_from_pretrained(repo_or_path: str) -> PolicySpec:
    """Load config.json from a local dir or a HF repo id and derive the spec."""
    import os

    cfg_path = None
    local = os.path.join(repo_or_path, "config.json")
    if os.path.exists(local):
        cfg_path = local
    else:
        # look in the HF hub cache
        cfg_path = _find_hf_cached_config(repo_or_path)
    if cfg_path is None:
        # last resort: download just the config
        try:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(repo_or_path, "config.json")
        except Exception as e:  # pragma: no cover
            raise FileNotFoundError(f"could not locate config.json for {repo_or_path}: {e}")

    with open(cfg_path) as f:
        return policy_spec_from_config(json.load(f), repo=repo_or_path)


def _find_hf_cached_config(repo: str) -> str | None:
    import glob
    import os

    cache = os.path.expanduser("~/.cache/huggingface/hub")
    slug = "models--" + repo.replace("/", "--")
    hits = glob.glob(os.path.join(cache, slug, "snapshots", "*", "config.json"))
    return hits[0] if hits else None
