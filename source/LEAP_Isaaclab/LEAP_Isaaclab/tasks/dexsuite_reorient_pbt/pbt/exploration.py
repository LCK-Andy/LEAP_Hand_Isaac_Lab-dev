# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Scoped control of rl-games' *learned*, state-independent Gaussian logstd.

Integration: put ``ExplorationObserver()`` in the runner's ``MultiObserver`` (also
when PBT selection is disabled), and call ``close()`` in the runner's finally
block. Use ``continuous_a2c_logstd``, ``fixed_sigma=True``, identity sigma
activation, and entropy coefficient 0 in the YAML. Entropy is not changed here.

Verified against installed rl-games 1.6.5: A2CAgent calls after_init BEFORE
Runner restores weights and applies its optional sigma override. We initialize
only the fresh model, wrap instance restore to project donor sigma (not reset
it), project before action inference, and use an optimizer post-step hook to
project after EVERY minibatch update, including GradScaler-driven steps.
No model forward or action/log-probability output is modified. Projection is
outside autograd, so stored Gaussian log-probabilities still describe the raw
samples PPO trains on, not the subsequently clipped environment commands.

Default std .3 with bounds [.05, 1] is a conservative, testable intervention,
not a measured optimum: it avoids the observed runaway std >50 while retaining
nonzero exploration. At zero mean .3 gives ~0.086% *analytic* coordinate clipping
at +/-1 (not measured rollout behavior). If an exact, action-ordered 23-joint
KUKA/Allegro name list is explicitly supplied, the arm gets initial .15 and cap
.5; fingers retain .3 and cap 1. The longer arm's joint motion moves the hand
farther, motivating the smaller exploratory command. Never assume the first
seven DOFs are the arm or use unordered robot names. Unknown mappings fall back
to the uniform profile and global diagnostics.
"""

import logging
import math
import torch

from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from rl_games.common.algo_observer import AlgoObserver

_LOG = logging.getLogger(__name__)
_MISSING = object()


class ExplorationObserver(AlgoObserver):
    """Bound logstd at safe lifecycle boundaries; own no global class patches.

    Args:
        initial_std: Fresh-model std only; restored in-range values survive.
        min_std: Positive lower exploration bound.
        max_std: Upper exploration bound.
        action_clip: Symmetric raw-policy command limit, normally 1.0. This is
            diagnostic only, and must match the runner/wrapper clipping setup.
        action_joint_names: Verified action-coordinate names, not asset order
            unless the action manager explicitly confirms they coincide.
        arm_initial_std: Fresh arm std when the exact joint mapping is known.
        arm_max_std: Arm upper bound, further limited by max_std.
    """

    def __init__(
        self,
        initial_std=0.3,
        min_std=0.05,
        max_std=1.0,
        action_clip=1.0,
        action_joint_names=None,
        arm_initial_std=0.15,
        arm_max_std=0.5,
    ):
        values = (
            initial_std,
            min_std,
            max_std,
            action_clip,
            arm_initial_std,
            arm_max_std,
        )
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Exploration standard deviations and action_clip must be finite and positive")
        if not min_std <= initial_std <= max_std:
            raise ValueError("Require min_std <= initial_std <= max_std")
        self.initial_std = initial_std
        self.min_std = min_std
        self.max_std = max_std
        self.action_clip = action_clip
        self.action_joint_names = tuple(action_joint_names) if action_joint_names is not None else None
        self.arm_initial_std = arm_initial_std
        self.arm_max_std = arm_max_std
        self.algo = None
        self._patches = {}
        self._step_hook = None

    def after_init(self, algo):
        """Attach once to an algorithm whose model and optimizer already exist."""
        if self.algo is not None:
            raise RuntimeError("ExplorationObserver is already attached; close it before reusing it")
        model = algo.model
        net = getattr(model, "a2c_network", None)
        sigma = getattr(net, "sigma", None)
        if not isinstance(model, ModelA2CContinuousLogStd.Network):
            raise TypeError("Exploration requires the continuous_a2c_logstd Gaussian model")
        if (
            not getattr(net, "fixed_sigma", False)
            or not isinstance(sigma, torch.nn.Parameter)
            or not sigma.requires_grad
            or sigma.ndim != 1
            or sigma.numel() == 0
            or not isinstance(getattr(net, "sigma_act", None), torch.nn.Identity)
        ):
            raise TypeError("Exploration requires a learned 1-D fixed_sigma parameter and identity sigma activation")
        if not any(sigma is p for group in algo.optimizer.param_groups for p in group["params"]):
            raise ValueError("The logstd parameter must belong to the policy optimizer")
        self._sigma = sigma
        self._low = torch.full_like(sigma, math.log(self.min_std))
        self._high = torch.full_like(sigma, math.log(self.max_std))
        self._initial = torch.full_like(sigma, math.log(self.initial_std))
        self._groups = self._joint_groups(sigma.numel())
        if "arm" in self._groups:
            arm_max = min(self.arm_max_std, self.max_std)
            if not self.min_std <= self.arm_initial_std <= arm_max:
                raise ValueError("Require min_std <= arm_initial_std <= min(arm_max_std, max_std)")
            arm = self._groups["arm"]
            self._high[arm] = math.log(arm_max)
            self._initial[arm] = math.log(self.arm_initial_std)
        self._projected_coordinates = sigma.new_zeros(())
        self._nonfinite_coordinates = sigma.new_zeros(())
        self._sample_clips = torch.zeros_like(sigma, dtype=torch.int64)
        self._mean_saturations = torch.zeros_like(sigma, dtype=torch.int64)
        self._sample_rows = 0
        self.algo = algo
        try:
            self._step_hook = algo.optimizer.register_step_post_hook(self._after_optimizer_step)
            with torch.no_grad():
                sigma.copy_(self._initial)
            self._wrap("restore", self._restore)
            self._wrap("get_action_values", self._get_action_values)
        except Exception:
            self.close()
            raise
        _LOG.info(
            "Controlled exploration attached: std initial=%s, bounds=[%s, %s], groups=%s; "
            "restored std is preserved inside bounds, projected otherwise",
            self.initial_std,
            self.min_std,
            self.max_std,
            list(self._groups),
        )

    def _joint_groups(self, size):
        names = self.action_joint_names
        arm = {f"iiwa7_joint_{i}" for i in range(1, 8)}
        fingers = {f"{finger}_joint_{i}" for finger in ("index", "middle", "ring", "thumb") for i in range(4)}
        if names is None or len(names) != size or size != 23 or set(names) != arm | fingers:
            if names is not None:
                _LOG.warning("Unverified KUKA/Allegro action mapping; using uniform exploration/global diagnostics")
            return {"global": list(range(size))}
        return {
            "global": list(range(size)),
            "arm": [i for i, name in enumerate(names) if name in arm],
            "fingers": [i for i, name in enumerate(names) if name in fingers],
        }

    def _wrap(self, name, callback):
        original = getattr(self.algo, name)
        previous = self.algo.__dict__.get(name, _MISSING)

        def wrapped(*args, **kwargs):
            return callback(original, *args, **kwargs)

        self._patches[name] = (previous, wrapped)
        setattr(self.algo, name, wrapped)

    @torch.no_grad()
    def _project(self):
        # Keep counters on the policy device; no per-minibatch GPU/CPU sync.
        finite = torch.isfinite(self._sigma)
        bounded = torch.maximum(torch.minimum(self._sigma, self._high), self._low)
        bounded = torch.where(torch.isnan(bounded), self._initial, bounded)
        self._projected_coordinates.add_((self._sigma != bounded).sum())
        self._nonfinite_coordinates.add_((~finite).sum())
        self._sigma.copy_(bounded)

    def _after_optimizer_step(self, optimizer, args, kwargs):
        self._project()

    def _restore(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        before = self._sigma.detach().clone()
        self._project()
        _LOG.warning(
            "Controlled exploration checkpoint intervention: projected %d/%d logstd coordinates; "
            "resulting std range [%g, %g] (in-range donor values retained)",
            int((before != self._sigma).sum()),
            self._sigma.numel(),
            float(self._sigma.detach().exp().min()),
            float(self._sigma.detach().exp().max()),
        )
        return result

    def _get_action_values(self, original, *args, **kwargs):
        # Also catches Runner._override_sigma, which executes AFTER restore.
        self._project()
        result = original(*args, **kwargs)
        with torch.no_grad():
            actions = result["actions"].detach().reshape(-1, self._sigma.numel())
            means = result["mus"].detach().reshape_as(actions)
            self._sample_clips.add_((actions.abs() > self.action_clip).sum(dim=0))
            self._mean_saturations.add_((means.abs() >= self.action_clip).sum(dim=0))
            self._sample_rows += actions.shape[0]
        return result

    @torch.no_grad()
    def metrics(self, reset=False):
        """Return windowed *sampled* clipping, not a Gaussian-tail estimate.

        Fractions count raw action coordinates that would be clipped at the
        configured symmetric limit, immediately before rl-games preprocessing.
        They do not measure downstream actuator/joint-limit saturation. Empty
        windows omit the fractions rather than claiming a measured zero.
        Conversion to host scalars is restricted to this reporting boundary.
        Projection counters are cumulative; sample counters reset only on request.
        """
        if self.algo is None:
            return {}
        result = {
            "exploration/projected_coordinates_total": float(self._projected_coordinates),
            "exploration/nonfinite_logstd_coordinates_total": float(self._nonfinite_coordinates),
        }
        std = self._sigma.exp()
        for name, indices in self._groups.items():
            prefix = f"exploration/{name}/"
            group_std = std[indices]
            result[prefix + "std_min"] = float(group_std.min())
            result[prefix + "std_max"] = float(group_std.max())
            result[prefix + "std_mean"] = float(group_std.mean())
            count = self._sample_rows * len(indices)
            result[prefix + "sampled_coordinates"] = count
            if count:
                result[prefix + "sampled_clip_fraction"] = int(self._sample_clips[indices].sum()) / count
                result[prefix + "mean_saturation_fraction"] = int(self._mean_saturations[indices].sum()) / count
        if reset:
            self._sample_clips.zero_()
            self._mean_saturations.zero_()
            self._sample_rows = 0
        return result

    def after_print_stats(self, frame, epoch_num, total_time):
        """Log to the usual rl-games writer once per statistics window."""
        if self.algo is None:
            return
        writer = getattr(self.algo, "writer", None)
        for name, value in self.metrics(reset=True).items():
            if writer is not None:
                writer.add_scalar(name, value, frame)

    def close(self):
        """Idempotently detach hooks; do not overwrite a subsequent owner's patch."""
        if self._step_hook is not None:
            self._step_hook.remove()
            self._step_hook = None
        if self.algo is not None:
            for name, (previous, installed) in self._patches.items():
                if self.algo.__dict__.get(name) is installed:
                    if previous is _MISSING:
                        delattr(self.algo, name)
                    else:
                        setattr(self.algo, name, previous)
        self._patches.clear()
        self.algo = None
