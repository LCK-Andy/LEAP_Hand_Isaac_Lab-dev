# FlashSAC continuation stopped for original DexPBT — 2026-09-28

The user explicitly requested stopping `flashsac-weste-20260928-dexsuite-r1-extend100m/prod` and then training original DexPBT on the Weste RTX 6000D. The named FlashSAC job has been stopped and preserved. The user then chose the original-stack route; the resulting native DexPBT launch is recorded in [the native run report](2026-09-28-native-dexpbt-weste.md).

## Verified graceful stop

- Remote: `root@connect.weste.seetacloud.com:26065`, NVIDIA RTX 6000D, 85,651 MiB.
- Exact process: PID 676498, argv matched `production_launch.json`, process-start ticks matched immediately before signalling.
- Actual entrypoint: `flashsac_dexsuite_20260928_weste_r1_extend100m/production_restore_check.py`; this wrapper runs the frozen FlashSAC trainer.
- Signal: SIGINT, handled at the trainer's vector/update boundary. Exit code **0**; `stopped_by_signal=true`.
- Final frame **87,370,752**, learner updates **170,250**, vector steps **85,323**.
- Completed episodes **247,680**, recorded successes **0**.
- Final checkpoint size **55,491,023 bytes**, SHA-256 `1b7991ee9ea21cbff688a56d991b13306afe7072cb42e05727ee1424df37517a`; ZIP CRC passed.

The continuation added 37,369,856 transitions to the original 50,000,896-transition run before the requested stop. It did not complete the additional100M budget.

Final checkpoint:

`/autodl-fs/data/flashsac_dexsuite_20260928_weste_r1_extend100m/train/checkpoints/latest.pt`

Preservation directory, containing before/final checkpoint hardlinks and source/launch/status records:

`/autodl-fs/data/flashsac_dexsuite_20260928_weste_r1_extend100m/stop_for_original_dexpbt_20260928T042950Z`

Local stop record: `logs/dexpbt_original_validation/remote_stop/stop_result.json` (ignored).

A first signalling attempt stopped before sending a signal because this remote Python lacks `os.pidfd_open`. The subsequent attempt revalidated argv and process-start identity, then used PID-specific SIGINT. No broad process-name kill or tmux-server termination was used.

## Other jobs preserved

Stage B PID465366 and Stage A PID470984 retained their exact argv and remained alive. Their `tdmpc2-stageb-train-r1` / `tdmpc2-stagea-train-r1` tmux sessions remained present. TensorBoard and all previous logs/checkpoints were left intact.

After stopping FlashSAC: approximately8,512MiB GPU memory used,76,531MiB free. GPU activity remained100% from the unrelated jobs; free memory does not imply exclusive compute access. Container memory approximately56.5GB of98.8GB at the observation.

## DexPBT launch decision

Remote installed environment: Isaac Lab3.0.0rc1, Isaac Sim6.1, Torch2.11/cu128, RL-Games1.6.5. Neither `isaacgym` nor `isaacgymenvs` is installed in that environment. A bounded directory search in `/root` and `/autodl-fs/data` found no original SDK/source checkout; this is not a proof against arbitrary hidden locations.

The current modern-port task is `Isaac-DexPBT-Allegro-Kuka-Reorientation-Direct-v0`, with the original recurrent PPO/PBT recipe. However, [the measured signed-inertia mismatch](2026-09-28-original-dexpbt-reproduction.md) prevents calling it a faithful physical reproduction. Its explicit `--allow-physics-mismatch` guard has not been bypassed. The original SDK route would require a separate compatible installation and Blackwell execution validation, without changing the working Isaac Lab stack.

At that decision point no replacement run was launched. The user subsequently chose **original-stack reproduction**. A separately validated original Isaac Gym/Python3.8/PTX-enabled Torch stack was prepared, and a fresh native8×8192DexPBT population was launched. See [native run and acceptance](2026-09-28-native-dexpbt-weste.md). The experimental Isaac Lab mismatch guard was not bypassed.
