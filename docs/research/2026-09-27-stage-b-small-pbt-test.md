# Stage B small-population PBT execution test

Date: 2026-09-27. Local PhysX validation requested during review of the Stage A/B and improved shared PBT changes.

## Configuration

- Task: `Isaac-Grasp-Motion-Cube-LEAP-Direct-v0`.
- Two members, eight environments each, seeds42/43.
-40 PPO epochs/member, horizon16:5,120frames/member and10,240aggregate frames.
- Native10-second episodes,148observations,16actions,unchanged grasp/motion/force gates and physics.
- `grasp_motion` objective and selection enabled; diagnostic publication/warmup/adaptation512frames,minimum4completed episodes. These shortened settings were test-only.
- Eager model execution; strict Stage A policy-only warm-start from member002/epoch192 donor and original manifests.

The improved shared observer/diagnostics are exercised. The rolling timestamped episode window is **specific to the `dexpbt` objective**; Stage B retains its existing disjoint `grasp_motion` windows. This test does not claim the rolling algorithm was applied to Stage B.

## Results

All four simulation processes completed explicitly with exit0 in137.96seconds total:two fresh members,one full resume,and the stationary smoke. Each member naturally completed16episodes.

| Member/window | Safe retention | Safe tracking | Overload incidence |
|---|---:|---:|---:|
|000 first8episodes|100%|0%|0%|
|001 first8episodes|100%|0%|0%|
|000 final8episodes|0%|0%|12.5%|
|001 final8episodes|25%|0%|0%|

Two native donor-selection calls returned `None`:neither saw a simultaneously valid peer window. **No exploitation or replacement was exercised.** These results demonstrate runnable training/selection plumbing,not acquisition or preserved learned retention after PPO updates.

## Transfer, resume and physical checks

- Actual124→148warm-start transfer was bit-exact for initial action means,sigmas and critic values. Optimizer/frame/epoch/best history started fresh.
- Member000 full-resumed epoch40/frame5,120 to epoch41/frame5,248,without reapplying warm-start. Adam steps200→205 and normalizer counts advanced; recipient RNG,lineage,effective parameters and separate best history were preserved. Evaluation evidence cleared.
- Checkpoint CPU audits passed:17model tensors,148-input shared256/128/64MLP,16actions,finite model/optimizer/inference.
- Stationary smoke:4env×330steps,finite148-d observations/rewards,99.697%valid grasp occupancy and0%tracking.
-51CPU regression tests plus2subtests passed.

No task source,remote jobs or installed packages changed during the test. No owned simulator process remained afterward. The new,unused Dexsuite policy-prior option was added by a parallel implementation before the resume; per-launch hashes record that change. Stage B/PBT source remained unchanged across the test.

## Local evidence

[Results and exact commands/source hashes](../../logs/stageb_pbt_sanity_20260927_014217/results.json) live with logs,configuration manifests,archived PBT publications and checkpoints. These are ignored local artifacts,not distributed with the repository. The tested Stage A donor likewise must be obtained separately with its manifests; see [Stage B commands](../stage-b-grasp-motion.md).

Checkpoint SHA-256:

- Member000 epoch40: `6fc0e3db3cbf8b26847b6e6f4fc54d6a798112fe09658ab64c9cce5bf74792f1`
- Member001 epoch40: `79e5a0ca0dac6a1752833de301210cbac210c9a660093a855a3ce92064390198`
- Member000 epoch41: `5325e313a80ef3ee0fbbc3454de0bce6bcd8975c300f0ca2e463ff4301557448`
