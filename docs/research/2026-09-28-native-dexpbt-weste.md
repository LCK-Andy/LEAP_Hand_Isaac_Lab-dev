# Native original DexPBT on Weste RTX 6000D

## User authorization and isolation

On 2026-09-28 the user requested stopping the named FlashSAC continuation and training DexPBT on Weste. After the known modern-PhysX inertia mismatch was explained, the user explicitly chose **original-stack reproduction**, not the experimental Isaac Lab port.

FlashSAC was gracefully stopped at 87,370,752 transitions and its final checkpoint preserved; see [stop record](2026-09-28-flashsac-stop-for-original-dexpbt.md). Stage A and Stage B TD-MPC2 jobs remain independent and must not be interrupted.

New isolated root on the data volume:

`/autodl-fs/data/dexpbt_original_native_20260928_weste_r1`

The validated `/root/autodl-tmp/env_isaaclab` environment is unchanged. The native route uses private Python3.8, original Isaac Gym Preview4, original IsaacGymEnvs source and separately installed learner dependencies. Private SDK assets/binaries are not published in this repository.

## Original runtime compatibility findings

A blanket claim that Isaac Gym Preview4 cannot run on Blackwell is contradicted by these supplied binaries and the actual remote tests:

- `libPhysXGpu_64.so` contains 63 forward-compatible PTX modules, not just old architecture cubins.
- Original **GPU PhysX** simulated four falling cubes for120steps on RTX6000D; their final centers were approximately0.027m above the plane.
- Original **GPU tensor pipeline** also completed120steps and exposed a CUDA root-state tensor descriptor of shape4×13.
- Both tests used original SDK code, without Isaac Lab or PyTorch. No viewer was created.

This establishes basic GPU/runtime support, not complete task/learning equivalence or performance.

### PyTorch compatibility

The existing Python3.8 Torch2.4.1+cu124 wheel contains no PTX in its main CUDA library; a basic CUDA operation on local sm120 failed with `no kernel image is available for execution on the device`.

An official surviving nightly provides a compatible candidate:

- `torch==2.5.0.dev20240727+cu124`, CPython3.8 wheel.
- Source build commit `092902bbd469df7bbc36b7f14dc16bac47f27084`.
- URL: `https://download.pytorch.org/whl/nightly/cu124/torch-2.5.0.dev20240727%2Bcu124-cp38-cp38-linux_x86_64.whl`.
- Wheel SHA256 `65723295d60a1125ece3920d53304c953f62a174b74e314ec0099dcdf31032ce`.
- Binary inspection confirmed339compute90PTX modules in `libtorch_cuda.so`.
- Real local Blackwell basic CUDA, MLP/Adam, and LSTM768 forward/backward/Adam tests passed.

The old runtime still prints an architecture warning because its native cubin architecture table predates sm120. Measured PTX execution, not suppression of the warning, is the acceptance evidence.

Matching `torchvision==0.20.0.dev20240727+cu124`, `pytorch-triton==3.0.0+dedb7bdf33`, NCCL2.21.5 and the wheel's exact CUDA dependency versions were installed privately. Original task NumPy aliases require NumPy1.23.5. Selected RL-Games1.6.0 is the original checkout's declared minimum; the paper did not publish an exact trainer lock, so this is not claimed to be a recovered paper-exact binary environment.

### Narrow runtime adaptation, not an MDP change

The first untouched task run reached the original117-d observations and LSTM construction, then its TorchScript random helper failed with NVRTC12.4's invalid `sm_120` architecture argument. `scripts/dexpbt_original/train_native.py --disable-gpu-fusion` uses `torch._C._jit_override_can_fuse_on_gpu(False)` before importing the task.

TorchScript remains enabled. The original source, equations, actions, observations, assets, physics and PBT algorithms are not rewritten. This disables an unsupported GPU fusion optimization; it is recorded as a runtime adaptation, not hidden as source parity.

The wrapper also pins the original Hydra config directory and prepends its isolated Python directory to PATH. This is necessary because the original PBT restarts using `os.execv(sys.executable, ["python3", ...])`; without PATH pinning it can escape the intended interpreter. Original `argv[0]` is the wrapper so the compatibility setting survives initial mutation and later exploitation. SDK/source/fusion settings propagate through process-local environment variables.

## Local native validation

- Original native task117observations/23actions loaded, no tactile/Dexsuite adapter.
- Two original recurrent PPO epochs at4environments/64minibatch completed and saved a full checkpoint.
- Full original checkpoint resume reached epoch3 from128frames, not a fresh learner.
- Native PBT initial mutation performed a real process re-exec, retained Python3.8 and GPU-fusion setting, then completed two PPO epochs.
- CPU wrapper regressions exercise four processes through two native re-execs, preserving config path/interpreter/member/seed/mutation settings.

Cold PTX compilation can take several minutes before the first training update; later iterations are distinct from first-launch compilation time. No claim of training throughput should be derived from the cold first epoch.

## Remote native training validation

The actual original learner completed the remote4-env/two-epoch smoke and wrote its full checkpoint. The first cold PTX pass took several minutes; this is initialization cost, not steady-state rollout throughput.

A paper-size single-member fit used **8192 environments, horizon16, minibatch32768, two PPO mini-epochs, original LSTM768**, with only a six-epoch execution budget. It passed:

- Six epochs /786432transitions; checkpoint46,204,787bytes, SHA256`c892bed691000cb24fadf1e00e4bfcf33d767c5c27b15dfc13a5dd9d3acbfa63`.
- Warm training throughput over epochs2–6:48,523 /59,994 /47,842 /64,902 /39,337transitions/s.
- Sampled own peakGPU6,550MiB; total container memory peak66.08GB, including existing jobs and file cache.
- Full native checkpoint resume advanced epoch6→7 and786432→917504transitions, retaining tolerance0.075. Resume checkpoint SHA256`ed5749b8c9b53715b087ce15b969d4f8ce3f713e7cc17649730a0997719580a9`.

These are short execution/restore measurements, not an acquisition result. Existing StageA/B were still running, so GPU compute was shared.

An additional original GPU-PhysX inertia lifecycle probe retained all six signed finger tensors through ten real GPU steps, with finite states. This closes the earlier CPU-only uncertainty for the basic robot configuration; it does not claim equality of complete CPU/GPU trajectories.

## Production launch and acceptance

A bounded **8×8192** pilot completed32epochs/member; all8 exited0 and all8 full checkpoints had finite model tensors,17optimizer state entries,4194304frames and tolerance0.075. Minimum free GPU memory21,506MiB. Container memory reached96.65GB of98.78GB, but included large reclaimable file caches; memory-high events occurred, **no max/OOM/OOM-kill events**. This is measured initial fit, not a promise against future memory growth.

Fresh production was launched2026-09-28 around13:32 UTC+8:

- Root: `/autodl-fs/data/dexpbt_original_native_20260928_weste_r1/prod8x8192`.
- Tmux sessions: `dexpbt-native-00` through `dexpbt-native-07`.
- Task: native `AllegroKukaLSTM task/env=reorientation`.
- Eight members,8192environments each, seeds42–49, original LSTM/PPO and reward mutation.
- Budget5,000,000,000transitions/member;40B aggregate. No pilot/FlashSAC weights reused.
- Original release PBT:20M publication interval,200M initial delay,100M post-restart adaptation,30% replacement fraction. This follows released code rather than silently substituting the paper-prose50M adaptation.
- Native RL-Games periodic checkpoint archive cadence is500epochs instead of source5000, providing earlier recovery; this changes checkpoint I/O, not PPO/PBT logic.
- Production source snapshot remains hash-verified861files with no source patches. Wrapper runtime adaptations are recorded and persist across native PBT re-exec.
- All eight members verified advancing over a common65-second interval, with no matching traceback/CUDA/OOM/buffer-overflow errors.
- Acceptance reached approximately32–37epochs/member,4.19–4.85Mcompleted transitions/member. Initial PBT mutation/re-exec worked; population exploitation is not expected before200M/member and is not yet established.
- Native scalar throughput in the simultaneous population was approximately9–14ktransitions/s/member in the first shared window; the common65.018-second log-counter sample measured86,685aggregate transitions/s; later single-iteration spikes are not steady-state measurements. The eight members share one GPU with StageA/B, so this does not reproduce eight-dedicated-V100 wall time.
- GPU at final observation63,537MiB used/21,506MiB free. Container95.49GB/98.78GB, including60.63GB file-accounted memory (21.84GB shared memory); no OOM. StageA PID470984 and StageB PID465366 remained alive, FlashSAC PID676498 absent.

Existing TensorBoard6007 was not restarted. A new logdir link exposes all eight scalar-bearing runs under:

`dexpbt-native-weste-20260928-8x8192`

Original PBT logging defers summaries240seconds, then uses interval-limited scalar writes; initial empty event files were not treated as dead training. All eight runs were verified through `/data/runs` and parsed event scalars. Initial mean successes remain approximately0–0.000244 and tolerance0.075; these are early exploration readings, **not acquired manipulation**.

Authoritative production records: `launch_record.json`, `progress_acceptance.json`, `tensorboard_acceptance.json`, `production_acceptance.json`. No automatic retry, shutdown or monitoring automation was created. The eight training sessions continue after acceptance.

## Remote evidence locations

Under the isolated root, `preflight/` holds source/archive hashes, exact dependency lock, logs and acceptance results. The pristine source snapshot is in `source/`, original SDK in `sdk/`, private Python in `python38/`, and learner packages in `env/`. Extension compilation uses `torch_extensions/`; driver PTX artifacts use `cuda_cache/`.

The source archive SHA256 is `c608ebda90a63a52929732f949b45d409bb36e337ea68192ec108aaa4ebc96ae`, derived from clean IsaacGymEnvs commit `665c32170d84b4be66722eea405a1e08b6e7f761` with no source patches. The source archive excludes `.git`; the provenance hash, not a fabricated new commit, identifies this deployment.
