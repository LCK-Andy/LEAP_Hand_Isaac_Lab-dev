# Persistent PBT randomness and historical best — 2026-09-25

## Dedicated mutation randomness

Each recipient now owns a dedicated `random.Random` stream. PPO/global reseeding does not change it. Normal PBT YAML publications contain `pbt_rng_state`, and staged replacement metadata saves the recipient's state **after** mutation. A restarted recipient restores its own stream, not the donor's; deterministic donor ranking is unchanged.

Exact continuation requires the companion metadata. Legacy metadata or a raw PPO checkpoint without that state gets an explicit warning and a reproducible, versioned fallback based on seed, member identity and recipient training age. It cannot recover the missing historical stream. Future publications persist the fallback state normally.

## Historical best and branch best

`BestCheckpointObserver` runs before the PBT callback can restart the process. It uses raw completed-episode reward, not the PBT selection objective. It retains:

- A recipient-owned historical best across all policy branches.
- A separate best for each branch created by exploitation.
- The same branch on ordinary checkpoint continuation.

Stable storage is `<log_root>/<pbt.workspace>/<member_id>/best/`, independent of timestamped per-process log directories. `best_history.json` records score, frame, epoch, branch and checkpoint provenance. Current `nn/historical_best.pth` and the conventional `nn/<config.name>.pth` reference historical best rather than the first completed episode after a restart.

The manifest is the commit authority. Checkpoints are written as distinct generations, then committed and aliases repaired. Garbage collection removes only owned generations not referenced by a branch winner or historical winner, so storage does not grow on every reward improvement. Per-branch winners remain retained.

For pre-fix history in an old timestamped recipient directory, explicitly provide:

```text
--legacy-best-checkpoint /absolute/path/to/recipient/old_run/nn/allegro_hand.pth
```

This must identify the recipient's known conventional best, not an arbitrary donor/latest checkpoint. Existing committed history takes precedence. A finite embedded score can seed the new ledger, but a historical checkpoint already overwritten by the old bug cannot be reconstructed. Provenance records this limitation.

## Verification

- Full simulator-free suite: **91 passed plus 7 subtests**. One expected warning exercises legacy RNG fallback.
- Focused formatting/lint and `git diff --check` passed. Pyright is unavailable locally; repository-wide lint contains unrelated pre-existing violations.
- Actual local Isaac/PPO smoke: three updates produced RNG-bearing publications and a historical-best manifest.
- Actual restart into a new timestamped log directory: resumed epoch 4, preserved the same RNG state when no mutations occurred, and retained the stronger epoch-2 historical best despite lower restarted reward. Both log-directory aliases resolved correctly.
- Regression tests cover uninterrupted versus resumed mutation sequences, recipient/donor isolation, legacy behavior, weaker branch protection, stronger promotions, interrupted checkpoint/manifest/alias operations, branch identity, and generation retention.

No remote training files or processes were modified, and no commit was made. Running remote processes will not acquire these changes until an explicit deployment/restart; these fixes do not remove the cold-episode reporting dip discussed separately.
