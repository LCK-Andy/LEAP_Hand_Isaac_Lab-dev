# Weste Dexsuite restart: 15 members × 128 environments

Date: 2026-09-27. User requested restarting from the current best checkpoint as a prior, with15members and128environments/member, using the improved PBT implementation.

## Run and source

- Task:`Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0`,PhysX.
- Remote workspace:`/root/autodl-tmp/dexpbt_reorient_20260927_weste_prior15x128/prod15`.
- Sessions:`dex-prior128-00` through `dex-prior128-14`.
- Immutable source:`a3d6b19d07af8533415ba13d765a884b1887bde2`.
-349committed files hash-verified;9LFS assets hydrated by expected object hashes;254ancillary asset files separately hash-proven from the previous immutable snapshot. No old Python code overlaid.
- Same curated EA experience and per-member Kit configuration isolation; no installed packages changed.

Configuration:15×128env,horizon32,seeds42–56,eager model,native initial hyperparameter diversity. **200,000epochs/member preserves819.2million transitions/member**, equal to the old512env×50,000epoch budget. Total population budget12.288billion transitions. This is a fresh optimization run,not continuation of old frame/optimizer state. Equal frame budgets do not imply equal learning or wall time with a different rollout batch.

Robot32position iterations and all task physics/gates are unchanged. PBT publication1Mframes,warmup5M,adaptation3M,min128completed episodes and production freshness/selection thresholds were not shortened.

## Prior selection

Selected member13/publication38,epoch2320/frame38,010,880,using eligible fresh physically matched evidence,ranked success-first then pose progress:

-2successes/3,081completed episodes (`0.00064914`).
-Progress:`0.20998629169113223`.
-Applied zero gravity/observation noise,training curriculum tolerance0.2,matching native initial physical conditions.

Empty or expired evaluations were unranked,not interpreted as quality zero. Raw-reward best-history files were inspected but not treated as the best PBT evidence. This is the top usable checkpoint at selection time,not a claim of solved behavior or statistical superiority from two successes.

Prior path:

`/autodl-fs/data/dexpbt_reorient_20260927_weste_prior15x128/prior/prior.pth`

SHA-256:`9b9ed468ce6f0c3cca4ae0c58c49bb2519980309a61144e5edb09b702bfeb25d`

The new [policy-prior adapter](../dexpbt-policy-prior.md) restores exact model and input/value normalizers only. All15start fresh optimizer,counters,environment/curriculum,best histories,evaluation evidence and recipient RNG. Distinct seeds retain bounded population diversity.

## Acceptance and controlled switchover

Actual-donor CPU acceptance on the remote stack proved all23model/normalizer states exact and identical128-row means,values and sigmas. Learned std0.23648–0.27323fit the unchanged0.05–1.0bounds,so no clamping was required.

There was insufficient safe GPU headroom for an additional simulator alongside the old15. After source/CPU checks,all old latest publication pairs and raw-best histories/configurations were hash-preserved before stopping only verified old trainers with SIGINT. All exited130without SIGTERM/SIGKILL. A post-stop reconciliation captured final in-flight publications as well. Original logs,source and checkpoints remain intact.

Preservation directories on the data volume:

-`/autodl-fs/data/dexpbt_reorient_20260927_weste_prior15x128/old_final_preservation`
-`/autodl-fs/data/dexpbt_reorient_20260927_weste_prior15x128/old_post_stop_latest`

On the freed GPU,the native128-env prior smoke passed2epochs/8,192frames,exit0,finite checkpoint,1745observations/23actions and fresh state. Production then launched with25-second staggering and resource/previous-member-exit guards.

Remote focused tests:46passed,1skipped. The installed pytest capture startup initially segfaulted importing the remote readline extension; explicit `-p no:capture` avoided that unrelated harness failure. No environment patch was made. Local full suite before deployment:249tests+42subtests passed.

## Verified launch state

All15members advanced across two observations;all15TensorBoard scalar streams had multiple samples,correct prior hash and no fatal errors.

- GPU use74,675MiB;reported available10,368MiB.
- Container RAM75.08/92GiB.
- Checkpoint directory:`/autodl-fs/data/dexpbt_reorient_20260927_weste_prior15x128/prod15/pbt_workspace`,linked through the expected training path.
- TensorBoard6007preserved old links and added `/root/tf-logs/dexpbt-weste-20260927-prior15x128`.

**The first production1M-frame publication had not yet occurred at launch acceptance.** The128-env smoke and CPU/regression checks establish execution/prior correctness; they do not establish population exploitation,learning improvement or higher aggregate throughput. No automatic retry or new monitoring automation was created.

Authoritative remote records include `prod15/launch_acceptance.json`,prior provenance/CPU acceptance,source manifests and old-stop/preservation records under the new workspace. These live deployment artifacts are not distributed with the source repository.
