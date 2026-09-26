# Stage A: sixteen-member Weste restart

## Active run

User requested remote Stage A training, explicitly Isaac Lab 3.0 EA, then 16 members and a restart visible in AutoDL TensorBoard. No credentials are recorded here. SSH key authentication worked; the supplied password was not needed.

- Host: `ssh -p 26065 root@connect.weste.seetacloud.com`
- GPU: NVIDIA RTX 6000D, 85,651 MiB reported VRAM.
- Container memory: **92 GiB hard cap / 90 GiB high threshold**, despite `free` displaying host-wide 754 GiB. Use cgroup memory, not host RAM, for sizing.
- Interpreter: `/root/autodl-tmp/env_isaaclab/bin/python`
- Isaac Lab: `3.0.0rc1` (validated EA package); Isaac Sim/kernel/PhysX cache: `6.1.0.0`; torch `2.11.0+cu128`; Warp `1.16.0`; Newton `1.5.2`; RL-Games `1.6.5`.
- Task: `Isaac-Grasp-Cube-LEAP-Direct-v0`, PhysX, unchanged nominal pregrasp/full-force MDP.
- **16 PPO/PBT members**, seeds 42–57, **512 envs/member**, 8,192 total concurrent envs.
- Horizon 16, minibatch 4,096, five mini-epochs, actor compilation disabled. Bounded exploration remains std .15 initially / [.03,.3], entropy zero.
- Budget: 5,000 epochs, 40,960,000 transitions/member, **655,360,000 aggregate**. Not an automatic shutdown schedule.
- PBT enabled: comparison every 524,288 transitions/member; first replacement eligibility after 5,242,880 transitions; post-replacement adaptation 3,145,728 transitions. Compares support-free retention then valid occupancy, with overload veto and matching physical conditions.
- Run root: `/root/autodl-tmp/leap_grasp_stage_a_20260926/training_16members_r1`
- Sessions: `grasp-stage-a-00` through `grasp-stage-a-15`.
- Member logs: `<run-root>/member_<i>.log`.
- Population checkpoints: `<run-root>/logs/rl_games/leap_grasp/pbt_workspace/{000..015}/`.
- RL-Games run dirs: `<run-root>/logs/rl_games/leap_grasp/member_<i>/`.

## AutoDL TensorBoard

The existing service on **port 6007**, serving `/root/tf-logs`, was left running. Added only a new symlink:

`/root/tf-logs/grasp-stage-a-16members-r1` → `<run-root>/logs/rl_games/leap_grasp`.

Verified HTTP `/data/runs` contains all 16 members and `/data/plugin/scalars/tags` exposes actual learning/grasp scalar tags for all 16—not merely empty directories. Select `grasp-stage-a-16members-r1/member_*/summaries` in AutoDL's TensorBoard UI; older pilot runs remain available separately.

Useful scalar tags:

- `Episode/GraspEpisode/retained`: support-free full-episode retention.
- `Episode/GraspEpisode/valid_fraction`: four-pulp-valid occupancy over the nominal horizon.
- `Episode/GraspEpisode/success_rate`: whether the continuous two-second hold was achieved.
- `Episode/GraspEpisode/overload_fraction`.
- `Episode/Grasp/support_force`, `Episode/Grasp/full_force_max`, `Episode/Grasp/hold_seconds`.

Direct SSH tunnel if needed:

```bash
ssh -p 26065 root@connect.weste.seetacloud.com -N -L 6007:127.0.0.1:6007
```

Then open http://127.0.0.1:6007 .

## EA runtime verification and isolation

Version metadata alone initially concealed an incomplete extension-cache upgrade: `isaacsim-extscache-kit` and `isaacsim-extscache-kit-sdk` still report 6.0.1.0. Initial startup stalled downloading extensions; registry-disabled startup exposed missing `omni.usdphysics` 110.3.2 and `omni.kit.usd.layers`.

Shared installation was **not overwritten or downgraded**. Deployed matching headless dependencies from the validated local Isaac Sim 6.1 installation:

- 51 extension directories, ~2.82 GiB, under `/root/autodl-tmp/leap_grasp_stage_a_20260926/ea_exts`.
- 2,818 overlay payload SHA256 checks passed. All 3,475 kernel files and 1,249 physics-cache files compared byte-identical to local.
- Curated 78-extension search view selects matched builds only, excluding old cache search paths.
- Custom experience: `/root/autodl-tmp/leap_grasp_stage_a_20260926/ea_runtime/isaaclab.python.headless.ea-curated.kit`.
- Registry and dependency-cache reuse disabled; member-specific user configs avoid old persistent settings.
- Scripts assert `isaaclab==3.0.0rc1`; `PYTHONPATH` points to isolated Stage A snapshot, not old task checkouts.
- Source snapshot includes uncommitted Stage A implementation based on git `af946a0b45dc35f99494fb33f250e423f13e211a`. 563 source/script/test/asset hashes verified remotely.

Evidence files in the deployment root: `source_manifest.json`, `environment_validation.json`, `ea_overlay_payload_hashes.json`, `ea_overlay_verified.json`, `smoke_curated.json`, `autodl_runs_verified.json`, `autodl_scalar_tags_verified.json`.

## Validation and restart history

Remote EA PhysX fixture smoke passed: four envs, 330 control steps, 124-d observations, 99.7% valid occupancy, zero drops/overloads/non-pulp support, full normal+friction signals, real contact-loss/reset and one-target-update-per-control-step checks. This is **fixture validation, not trained-policy performance**.

Remote simulator-free logic tests: 31 tests +27 subtests passed; two additional environment-contract unittest tests passed. A conda readline crash under the inherited terminal/locale was avoided with process-local `LC_ALL=C TERM=xterm`, without editing system libraries.

Pilot artifacts preserved:

- `pilot_01`: command-line quoting failure, no training.
- `pilot_02`: four-member 1,024-env concurrent pilot completed, real grasp publications and checkpoints.
- `pilot_03`: sixteen compiled members; one exited 137 and cgroup high-memory pressure observed. No CUDA OOM reported. Coordinator stopped only these pilot sessions and retained outputs.
- `pilot_04`: staggered/eager sixteen-member attempt; later inspection found no remaining sessions. Do not infer completion from absent `.exit` files.
- `pilot_04_eager`: separately present eight-member pilot artifacts; latest inspection found no running sessions. Preserved rather than overwritten.
- User-requested restart uses **fresh** `training_16members_r1`; it does not mix partial pilot checkpoints into the population. Reduced envs/member from1,024 to512 to leave host-memory headroom while honoring16 members. No source MDP changes.

Restart verification: all16 members showed advancing PPO epochs, no exits or traced training errors. At one concurrent snapshot: **68.1 GiB cgroup memory**, **47,993 MiB VRAM**,100% GPU busy-time; per-member epochs3–28 due staggered launches. Throughput and early reward are not learning-quality claims.

## Monitoring and manual control

```bash
# Read-only per-member progress/checkpoint summary
LC_ALL=C TERM=xterm /root/autodl-tmp/env_isaaclab/bin/python \
  /root/autodl-tmp/leap_grasp_stage_a_20260926/status.py training_16members_r1

# Attach one member (detach with Ctrl-b then d)
tmux attach -t grasp-stage-a-00
```

`restart_16.py` and `<run-root>/launch_record.json` preserve exact argv and environment. The restart script refuses an existing run root; do not rerun it as a resume command. Resume an individual member from its own numbered/full checkpoint plus companion metadata, preserving member identity and accounting; simulator state and partial episode windows are not checkpointed. Conventional reward-best files are not guaranteed grasp-objective champions. No automatic crash retry or machine shutdown is configured.
