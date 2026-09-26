# Restart Dexsuite PBT from a policy prior

`train_dexpbt_member.py --policy-prior PATH` initializes a **fresh**
`Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0` run from a compatible checkpoint.
It supports changing the environment count, for example512 to128, without
restoring incompatible per-environment curriculum state.

## What is restored

- Complete separate actor/critic512/256/128ELU model,1745observations and23actions.
- Learned raw logstd and both input/value normalizers, exactly and without padding.

Optimizer,frame/epoch counters,environment/curriculum,best-history,episode/evaluation
windows and RNG remain those of the new recipient. This is not continuation of the
old training trajectory. The new run uses its native initial curriculum; a prior
trained under harder conditions does not implicitly restore those conditions.

## Required artifacts and checks

The trusted local checkpoint needs exactly one complete `agent.yaml`/`env.yaml`
manifest pair:beside the checkpoint,in its directory's `params/`,or in the usual
`<run>/params/` when the checkpoint is in `<run>/nn/`. Checkpoints and manifests
from local validation are not distributed with the repository.

The adapter validates architecture,observation/action ordering,and native MDP/physics
configuration. Allowed differences are launch bookkeeping,seed,device and environment
count. Exact scene namespace expansion before/after spawning is normalized; broader
contact-filter regexes are not treated as equivalent. Actual built observation groups
must be170/615/960channels and23actions. Nonfinite/incompatible weights,missing or
conflicting normalizers,and donor logstd outside current exploration bounds are refused.

Use a new workspace and experiment directory. `--policy-prior` cannot be combined
with full `--checkpoint`,PBT resume,Stage A/B `--warm-start`,legacy-best initialization,
sigma overrides or hidden `load_checkpoint/load_path`. Exploit restarts remove the
prior flag and use the selected donor's normal full-checkpoint restart path.

## Example

From the repository root,with a compatible donor and installed Isaac Lab environment:

```bash
DEXSUITE_PRIOR=/absolute/path/to/dexsuite/nn/checkpoint.pth
/home/andy/env_isaaclab/bin/python scripts/rl_games/train_dexpbt_member.py \
  --task Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0 \
  --headless --num_envs 128 --max_iterations 200000 \
  --policy-prior "$DEXSUITE_PRIOR" \
  pbt.directory=/absolute/path/to/fresh_population \
  pbt.num_policies=15 pbt.policy_idx=0 \
  params.config.full_experiment_name=member_0 \
  params.config.torch_compile=false
```

This launches **one member**. A15-member population needs one process per policy index
0–14 with distinct seeds/names and the same population directory. Machine-specific
curated EA experiences,storage location and resource limits still apply.

With horizon32,512env×50,000epochs and128env×200,000epochs both correspond to
819.2million transitions/member. Equal transition budgets do not imply equal learning
or wall time:the smaller rollout batch changes update cadence. The launch should record
which budget convention it uses.

## Validation scope

CPU tests cover exact model/normalizer output transfer to128-row batches,fresh optimizer
and RNG,manifest/architecture rejection,namespace handling,fresh destinations and restart
argument stripping. A real normally registered128-env pre-spawn config was checked against
a saved512-env donor manifest. A live simulator smoke is still required on each deployment
stack; CPU contract acceptance is not physics or learning validation.
