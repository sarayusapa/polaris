# SO-101 embodiment port

Porting PolaRiS (a DROID/Franka-locked real2sim eval harness) to the **SO-101**
arm (5 arm + 1 gripper = 6 Feetech STS3215 servos), so a user's own LeRobot
policy can be evaluated in a real2sim scene.

## Status

**Robot-side port is complete and validated in Isaac Sim** (each step tested
headless, not just exit codes):

| # | step | status | artifact |
|---|------|--------|----------|
| 1 | Locate SO-101 URDF + meshes | ✅ | `urdf/` (fetched, gitignored) |
| 2 | URDF → USD | ✅ | `convert_so101_urdf.py` → `PolaRiS-Hub/so101/so101.usd` |
| 3 | Spawn + validate (6 DOF, no crash) | ✅ | `validate_so101_usd.py` |
| 4 | `so101_robot_cfg.py` (actuators, home pose) | ✅ | `src/polaris/environments/so101_robot_cfg.py` |
| 5 | 6-dim continuous action (index→joint verified) | ✅ | `so101_cfg.py` (`ActionCfg`) |
| 6 | Wrist camera mounted on gripper | ✅ | `so101_cfg.py` (`SceneCfg`) |
| 8 | Rubric retargeted (gripper-agnostic) | ✅ | `rubrics/checkers.py`, `rubrics/so101_rubrics.py` |
| 7 | SO101 InferenceClient | ⏳ blocked | needs a trained policy's I/O contract |
| 9 | real2sim scene (splat + objects) | ⏳ blocked | needs captured scene (external `real2simeval` pipeline) |
| 10 | end-to-end eval | 🔒 | blocked on 7 + 9 |

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

## Remaining work (needs the user)

- **#7 InferenceClient** — needs the trained policy's exact I/O (image res /
  normalization, state format/units, action space/units, control freq/chunking).
- **#9 scene** — needs the captured table: splat reconstruction + object meshes +
  ChArUco-calibrated poses **in the SO-101 base frame** + `scene.usda` +
  `initial_conditions.json`.
- **#10** — run `eval.py` with a Fake policy first, then the real client.

### OSS policy for a harness test (parked)

To dry-run #7/#10 before the user trains their own, a good off-the-shelf policy
is `mot-prog/so101_pick_up_wrist_pan_act` on HF Hub (ACT; `observation.state`[6],
`action`[6]; cameras `wrist` + `shoulder_pan`). **Blocker:** it uses LeRobot's
new "processor" checkpoint format needing **lerobot ≥ 0.4**, but the only
installed lerobot is **0.1.0** (openpi-pinned). Cleanest path = run the ACT
policy in a small dedicated lerobot env as a websocket server (mirrors the π0.5
setup); the `SO101Client` connects to it. Decision deferred.
