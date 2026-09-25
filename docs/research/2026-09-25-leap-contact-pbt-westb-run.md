# LEAP contact-point PBT run — 2026-09-25

## Deployment and scope

User requested training the LEAP hand to reach contact points on the cube on the existing westb server. Launched native `Isaac-Contact-Cube-LEAP-Direct-v0` with PPO/PBT, PhysX, measured force gate0.1N, native randomized continuous hold requirement unchanged. This is not the orientation-only Allegro task or a geometry-only Newton fallback.

- SSH: `ssh -p 46718 root@connect.westb.seetacloud.com`
- Root: `/root/leap_contact_pbt_20260925`
- Frozen source: `code/`; exact commands/config: `population.json`; validation: `launch_validation.json`.
- Snapshot hash manifest: `snapshot_sha256.json`, SHA256 `c4951c0a2e736c1ec25306de3ad9bc947227003caca020348b54b1b87cdd6657`.
- Four members x2048 environments, seeds42–45, two members per GPU.
- 1526 PPO epochs, 50,003,968 transitions/member (~200M total).
- Publication524,288 steps; warmup5,242,880; post-copy adaptation3,145,728.
- Score: completed native held goals/episode; when both candidate windows have zero goals, full-horizon-normalized measured matched-contact occupancy is a progress tie-break. Margin0.001 for goals, separate0.00001 for sparse contact, one-standard-error guard, minimum512 completed episodes, freshness checks.
- Dedicated persistent PBT RNG and historical/branch-best protection included.

## Correctness prerequisites

Three task bugs were fixed before production: distance gather selected the wrong point column; geometry refresh during partial reset advanced unrelated hold timers; and successful point resampling left returned observation vectors describing the old goal. Independent regressions and review verified the corrections. Reward coefficients, actions, observation dimensions184, sampling, force gate and hold difficulty were not redesigned.

The inherited smoke used an effectively zero hold duration that could pass due to epsilon comparison even without real holding. Replaced by a controlled measured-force snapshot with .1s hold: no goal on the first two reward steps, confirmation on the third. This is a mechanics test, not evidence that the learned policy naturally succeeds.

## Validation and initial observation

- Deployed snapshot100 tests plus7 subtests passed.
- Remote8-env PhysX smoke: correct6cm cube, obs(8,184), live per-pulp maximum forces36.98/19.87/16.61/27.55N.
- PPO2048-env probe786,432 steps completed; resume continued to917,504 with optimizer, bounded sigma and final checkpoint metadata.
- Production all4 advancing at approximately819,200steps/epoch26 at final check. Every member published at524,288steps with2730–2957 completed episodes, RNG state and best-history records.
- Initial throughput10.2–10.5k transitions/s/member (~41.5k aggregate), ~7GB/32GB VRAM per GPU, root8.9GB free. Initial remaining estimate80minutes, subject to contacts and restart overhead.
- No observed OOM/contact-buffer overflow/patch overflow/traceback.
- Std~.293–.304, action clipping~.16–.20%. Natural held-goal successes still zero; contact occupancy nonzero but sparse. No acquisition claim.

## Monitoring and preservation

- tmux `leap-contact-population-0` through `leap-contact-population-3`.
- Logs `population/member_<i>.log`.
- Checkpoints `population/logs/rl_games/leap_contact/pbt_workspace/{000,001,002,003}`.
- Existing TensorBoard preserved on6007; new entry `/root/tf-logs/leap-contact-pbt-20260925`.
- Tunnel: `ssh -p 46718 root@connect.westb.seetacloud.com -N -L 6007:127.0.0.1:6007`, then http://localhost:6007.

Existing remote experiments and checkpoints preserved. No commit/push performed. Concurrent unrelated local PBT edits appeared during implementation; those were preserved locally and excluded from the frozen deployed snapshot rather than silently changing the validated run. Refer to snapshot hashes, not subsequent local HEAD, when diagnosing production behavior.
