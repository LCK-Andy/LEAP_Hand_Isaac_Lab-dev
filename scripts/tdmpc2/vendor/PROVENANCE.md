# Official TD-MPC2 source provenance

Vendored from the authors' official repository, https://github.com/nicklashansen/tdmpc2,
under the adjacent MIT LICENSE (Copyright Nicklas Hansen, 2023).

Canonical pin: `e9f59321933cbc8e11a002b842adc7d4ffae8ff1` (2026-07-13,
“fix q ensemble weight init”). Files retained from `tdmpc2/`:

- `tdmpc2.py`
- `common/world_model.py`
- `common/layers.py`
- `common/math.py`
- `common/scale.py`
- `common/init.py`

Exact source base:
https://github.com/nicklashansen/tdmpc2/tree/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2

Local changes inside vendored code are limited to qualified relative imports,
configurable device instead of hardcoded cuda:0, CPU-compatible Adam capturable
settings, and the following TensorDict 0.14 compatibility adaptation: the Ensemble's
meta functional template is nonregistered (the TensorDict parameters own all real
weights); `train()` explicitly propagates to the template. This avoids `.to()` trying
to materialize meta tensors and preserves dropout mode for online and target Qs.
The ensemble forward now flattens the registered TensorDict parameters to a named
mapping and uses `torch.func.functional_call` inside `vmap`. The prior
`params.to_module(self.module)` call rewrapped batched tensors as Parameters on
TensorDict0.14 and disconnected critic gradients. CPU and CUDA regressions now
require every critic to learn a known positive-return fixture, retain policy-action
gradients through detached critics, and keep target weights independent. This repairs
an integration defect; learned objectives, target updates, initialization and model
architecture remain unchanged.

The enclosing `agent.py` implements the official batched MPPI layout described in
`vectorized_env` pin `97c14471997d5e58420d05d0fd6a5bd0195410dd` (2025-05-21):
https://github.com/nicklashansen/tdmpc2/blob/97c14471997d5e58420d05d0fd6a5bd0195410dd/tdmpc2/tdmpc2.py

It deliberately does not copy that branch's collector: that collector asserts that
all environments reset together. Local planning adds asynchronous per-row reset masks,
GPU action return, and environment chunking. It retains canonical main's probability
Gumbel sampler: the vector branch calls its probability-taking helper with log(score),
which logs a second time and produces NaNs. Batching must not reproduce that bug.
The upstream eval_mode still samples candidates and an elite trajectory; it disables
only additional final action noise, and is not deterministic evaluation by itself.

`config.py` expresses the official 5M architecture and defaults; single-task
`task_dim=0`, `episodic=True` for real terminal events, `compile=False` for initial
compatibility validation, and the Isaac task's observation/action/episode sizes are
explicit integration choices. The official constructor adds two planning iterations
when action_dim >= 20, so base iterations=6 means eight for 23 actions.
Small test networks and reduced candidate counts are test fixtures, not production defaults.

No TorchRL dependency is imported by this vendor subset. The surrounding trainer supplies
sequence replay through the exact upstream sample tuple. Full learner checkpoints are
implemented by the wrapper because upstream save() stores only the world model.
Replay, simulator state, recipient PBT frame accounting, and checkpoint file ownership
remain the trainer's responsibility. Reset planner history after a simulator restart.

Validated simulator-free against Python 3.12, torch 2.11.0+cu128, TensorDict 0.14.0:
CPU batched planning, single-environment seeded parity against the canonical planner,
finite learner update, exact next-update checkpoint roundtrip, and reset-row isolation.
This is not evidence of Isaac simulation throughput, task learning, or GPU resource fit.
