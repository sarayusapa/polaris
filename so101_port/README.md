# SO-101 embodiment port

Porting PolaRiS (a DROID/Franka-locked real2sim eval harness) to the **SO-101**
arm (5 arm + 1 gripper = 6 Feetech STS3215 servos), so a user's own LeRobot
policy can be evaluated in a real2sim scene.

## Status

**Robot-side port is complete and validated in Isaac Sim** (each step tested
headless, not just exit codes):

| # | step | status | artifact |
|---|------|--------|----------|
| 1 | Locate SO-101 URDF + meshes | done | `urdf/` (fetched, gitignored) |
| 2 | URDF → USD | done | `convert_so101_urdf.py` → `PolaRiS-Hub/so101/so101.usd` |
| 3 | Spawn + validate (6 DOF, no crash) | done | `validate_so101_usd.py` |
| 4 | `so101_robot_cfg.py` (actuators, home pose) | done | `src/polaris/environments/so101_robot_cfg.py` |
| 5 | 6-dim continuous action (index→joint verified) | done | `so101_cfg.py` (`ActionCfg`) |
| 6 | Wrist camera mounted on gripper | done | `so101_cfg.py` (`SceneCfg`) |
| 8 | Rubric retargeted (gripper-agnostic) | done | `rubrics/checkers.py`, `rubrics/so101_rubrics.py` |
| 7 | SO101 InferenceClient | done | `src/polaris/policy/so101_client.py` |
| 10 | end-to-end eval (OSS ACT policy) | done | `SO101-FoodBussing` env, plumbing test |
| 9 | real2sim scene (splat + objects) | blocked | needs captured scene (external `real2simeval` pipeline) |

**Runs end-to-end** with an off-the-shelf LeRobot ACT checkpoint
(`mot-prog/so101_pick_up_wrist_pan_act`) served from a dedicated `lerobot` conda
env (`serve_lerobot_act.py`) into the `SO101-FoodBussing` env (SO-101 dropped in
the reused splat scene). This is a *plumbing* validation only — the arm is
outside the Franka-framed camera and the policy is out-of-distribution, so
`progress = 0.0`. A meaningful eval needs a real2sim scene built for the SO-101
(#9) + a policy trained for it.

### Generic embodiment framework

The SO-101 work above is generalized so a new robot is *"convert URDF → fill a
spec → point at a checkpoint → validate"* (`src/polaris/embodiment/`):

| module | what |
|--------|------|
| `spec.py` | declarative `EmbodimentSpec` (+ `specs/so101.yaml`) |
| `urdf_ingest.py` | any URDF → spec (verified: SO-101, Franka) |
| `policy_adapter.py` | any LeRobot checkpoint → I/O contract (verified: ACT, SmolVLA) |
| `builders.py` | spec → IsaacLab cfgs (app-gated) |
| `tasks.py` / `validate.py` | spec-driven rubric / `validate_embodiment(spec)` |
| `policy/lerobot_client.py` + `serve_lerobot.py` | generic client+server for any LeRobot policy |

GPU-free smoke tests: `test_embodiment_cpu.py`, `test_serve_lerobot_cpu.py` (pass).
Deferred (GPU): `test_spec_builders_gpu.py` (spec-built SO-101 == hand-written).

## Reproduce the robot asset

The SO-101 URDF + STL meshes are **not vendored** (third-party, gitignored).
Fetch and convert:

```bash
bash so101_port/fetch_urdf.sh                 # -> so101_port/urdf/ (from TheRobotStudio/SO-ARM100)
# in the `polaris` conda env, with polaris_env_conda.sh sourced:
python so101_port/convert_so101_urdf.py       # -> PolaRiS-Hub/so101/so101.usd
```

Validate / inspect (all require `AppLauncher(enable_cameras=True)` and `python -u`,
since Isaac Sim block-buffers stdout to files and hangs on shutdown — kill the
process after it prints its result):

```bash
python so101_port/validate_so101_usd.py       # 6 DOF, joint names, holds home pose
python so101_port/test_so101_cfg.py           # SO101 ArticulationCfg (actuators, home)
python so101_port/test_so101_action.py        # 6-dim action index->joint mapping
python so101_port/test_so101_camera.py        # wrist cam renders + mounted on gripper
python so101_port/test_so101_rubric.py        # ee_frame + gripper open/close detection
python so101_port/view_so101_live.py          # GUI viewer (headless=False), watch it move
```

## Design notes

- **Action space** = 6-dim **continuous joint position** in LeRobot motor order
  `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`
  (absolute radians). This differs from DROID's 7 joints + **binary** gripper.
- **Gripper convention is inverted** vs DROID: SO-101 gripper is OPEN at LARGE
  joint values, CLOSED at small ones. `is_within_xy` was made gripper-agnostic
  (`gripper_joint`, `open_is_large` params); `so101_rubrics.pick_place_rubric`
  wires it correctly.
- **Wrist camera** is anchored to `gripper_frame_link`. Its orientation offset is
  a **placeholder** — set it to match the real mounted camera before real eval.
- The base is a **fixed root** (`fix_base=True` at conversion, `fix_root_link`),
  i.e. table-mounted.

## Running the SO-101 eval (OSS harness)

1. Create the `lerobot` env once (lerobot ≥ 0.4 needs Python ≥ 3.12; the
   openpi-pinned 0.1.0 can't load modern "processor" checkpoints):
   ```bash
   conda create -y -n lerobot python=3.12
   conda run -n lerobot pip install "lerobot==0.6.0" websockets msgpack msgpack-numpy
   conda run -n lerobot pip install -e third_party/openpi/packages/openpi-client
   ```
2. Serve the policy (dedicated env) and eval (polaris env), in two shells:
   ```bash
   conda run -n lerobot python so101_port/serve_lerobot.py \
       --repo mot-prog/so101_pick_up_wrist_pan_act --port 8000 \
       --state-units degrees --action-units degrees
   # then, in the polaris conda env with polaris_env_conda.sh sourced:
   python scripts/eval.py --environment SO101-FoodBussing \
       --policy.client LeRobot --policy.port 8000 --run-folder runs/so101
   ```
   (`so101_client.py` + `serve_lerobot_act.py` are the hand-written equivalents,
   kept for reference; the generic `LeRobot` client + `serve_lerobot.py` above
   supersede them.)

## Remaining work

- **#9 real2sim scene** *(needs the user)* — the captured table: splat
  reconstruction + object meshes + ChArUco-calibrated poses **in the SO-101 base
  frame** + `scene.usda` + `initial_conditions.json` (external `real2simeval`
  pipeline). Required for a *meaningful* eval, plus a policy trained for it and
  the wrist-cam extrinsics set to the real mount.
- **Deferred (GPU)** — `test_spec_builders_gpu.py` (spec-built == hand-written);
  generic `build_env_cfg`/registration; a **camera reposition** so the SO-101 is
  actually in frame (the FoodBussing external cam is aimed at the Franka's ~0.5 m
  workspace, beyond the SO-101's ~0.35 m reach).
