# DexPBT scaled experiment — westb, 2026-09-24

## Deployment

- SSH: `ssh -p 46718 root@connect.westb.seetacloud.com` (credentials are not stored here).
- Hardware reports two NVIDIA GeForce RTX 4080 SUPER devices, **32,760 MiB each**, 128 available CPUs, approximately 500 GiB host RAM. GPUs were idle before deployment.
- Existing full Isaac environment: `/root/autodl-tmp/env_isaaclab`, Isaac Sim 6.0.1, Isaac Lab 6.1.18, Torch 2.10/cu128.
- Isolated code: `/root/dexpbt_scaled_20260924/code`; existing `/root/LEAP_Hand_Isaac_Lab` and other experiments are unchanged.
- Server has RL-Games 1.6.1; validated 1.6.5 is installed **only** in `/root/dexpbt_scaled_20260924/python_deps`. Member PYTHONPATH selects it without modifying the shared venv.
- Code hashes: `/root/dexpbt_scaled_20260924/code_manifest.json`.
- Exact commands/environment/GPU/seed assignments: `/root/dexpbt_scaled_20260924/experiment.json`.

## Population

Fresh corrected-policy population, not resumed from the old high-noise cqa1 runs:

- Eight members, four per GPU, each with 1,024 environments; 8,192 total environments.
- Members 0–3 use physical GPU 0; 4–7 use GPU 1, via `CUDA_VISIBLE_DEVICES`. Each process uses logical cuda:0.
- Seeds 42–49; bounded initial hyperparameter diversity; member 0 baseline.
- 10,000 PPO updates, horizon 32: nominal **327,680,000 transitions/member**, **2,621,440,000 aggregate**.
- Validated PPO settings unchanged other than environment count; 1,024 minibatch, five epochs. Entropy zero, std initial .3 bounded [.05,1].
- PBT publication every 1M transitions; 5M warm-up, 3M adaptation, minimum128 completed-episode samples, physical-condition matching.
- `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, separate Inductor cache/member. Startup staggered across both GPUs.
- Population workspace `/root/dexpbt_scaled_20260924/population`.
- tmux sessions `dexpbt-scaled-0` through `dexpbt-scaled-7`.
- Logs `population/member_<i>.log`; terminal exit statuses appear in `population/member_<i>.exit`.
- Scripts `/root/dexpbt_scaled_20260924/member_<i>.sh`; launcher `launch_population.py` refuses to replace existing population/sessions.

## Validation at launch

- Remote regression suite: **42 tests +7 subtests passed**. Server readline extension segfaults in pytest capture initialization; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ... python -m pytest -p no:capture -q .../tests` passes. No shared dependency changes were made to address this.
- Eight-environment simulator smoke passed initial tolerance, complete-episode measurements, fixed ADR0/1 and curriculum state restoration.
- Isolated 1,024-environment/12-update capacity probe passed with no contact-buffer overflow, about 6–7k total steps/sec solo after startup; early allocated VRAM approximately5–6GB.
- All eight production processes reached PPO updates; first PBT publication had a nonempty completed-episode condition group and effective optimizer parameters.
- Initial full-population throughput was roughly34.5k transitions/sec while the last member was still warming up. This is not a stable benchmark or learning-success claim.
- GPUs initially around23–24GB allocated each, leaving headroom; no Python traceback, CUDA OOM or contact-patch overflow observed in production logs at check.
- Isaac logs missing libXt.so.6 rendering-library warnings; headless smoke and training continue. Rendering has not been validated.

## Monitoring

Existing server TensorBoard was not restarted. Added `/root/tf-logs/dexpbt-scaled-20260924` pointing to this population's RL-Games logs; `/data/runs` verified discovery.

```bash
ssh -p 46718 root@connect.westb.seetacloud.com -N -L 6007:127.0.0.1:6007
```

Then open http://localhost:6007 and select runs under `dexpbt-scaled-20260924`.

Attach a member:

```bash
ssh -p 46718 root@connect.westb.seetacloud.com -t 'tmux attach -t dexpbt-scaled-0'
```

Launch validation does not establish successful task acquisition. PBT exploitation is intentionally deferred until the 5M-step warm-up and only occurs with sufficiently supported score differences; actual donor replacement has not yet been observed in this production run.
