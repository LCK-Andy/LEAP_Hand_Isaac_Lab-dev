"""Public, simulator-free curriculum and fixed-definition episode measurement.

Metrics are *training-condition* statistics unless a caller explicitly freezes
ADR and runs a separate held-out rollout. Success means at least one simultaneous
pose/contact match in a completed episode, not held goals or soft reward.
"""

import json
import math

FIXED_POSITION_TOLERANCE = 0.1
FIXED_ROTATION_TOLERANCE = 0.5
FIXED_CONTACT_THRESHOLD = 0.01


class SuccessToleranceCurriculum:
    """Anneal on consecutive, non-overlapping episode-count-weighted windows."""

    def __init__(
        self, start_std=0.2, final_std=0.1, factor=0.9, window_episodes=1024, required_windows=3, rate_threshold=0.5
    ):
        if not (0 < final_std <= start_std and 0 < factor < 1):
            raise ValueError("Invalid tolerance range or factor")
        if window_episodes < 1 or required_windows < 1 or not 0 <= rate_threshold <= 1:
            raise ValueError("Invalid curriculum window")
        self.start_std = start_std
        self.final_std = final_std
        self.factor = factor
        self.window_episodes = window_episodes
        self.required_windows = required_windows
        self.rate_threshold = rate_threshold
        self.tolerance = start_std
        self.pending = []
        self.streak = 0
        self.completed_episodes = 0

    def observe(self, successes):
        """Consume each completed environment's sticky flag exactly once."""
        values = [bool(value) for value in successes]
        self.completed_episodes += len(values)
        self.pending.extend(values)
        while len(self.pending) >= self.window_episodes:
            window = self.pending[: self.window_episodes]
            del self.pending[: self.window_episodes]
            self.streak = self.streak + 1 if sum(window) / len(window) > self.rate_threshold else 0
            if self.streak >= self.required_windows:
                self.tolerance = max(self.final_std, self.factor * self.tolerance)
                self.streak = 0
        return self.tolerance

    def get_state(self):
        """Return independent JSON-safe state, including partially filled windows."""
        return {
            "version": 1,
            "config": {
                key: getattr(self, key)
                for key in ("start_std", "final_std", "factor", "window_episodes", "required_windows", "rate_threshold")
            },
            "tolerance": self.tolerance,
            "pending": list(self.pending),
            "streak": self.streak,
            "completed_episodes": self.completed_episodes,
        }

    def set_state(self, state):
        """Restore only validated state with identical curriculum semantics."""
        if state.get("version") != 1 or state.get("config") != self.get_state()["config"]:
            raise ValueError("Incompatible curriculum checkpoint")
        tolerance = float(state["tolerance"])
        pending = list(state["pending"])
        streak = state["streak"]
        count = state["completed_episodes"]
        if not math.isfinite(tolerance) or not self.final_std <= tolerance <= self.start_std:
            raise ValueError("Invalid checkpoint tolerance")
        if len(pending) >= self.window_episodes or any(type(value) is not bool for value in pending):
            raise ValueError("Invalid pending episode window")
        if not isinstance(streak, int) or not 0 <= streak < self.required_windows:
            raise ValueError("Invalid successful-window streak")
        if not isinstance(count, int) or count < len(pending):
            raise ValueError("Invalid completed episode count")
        self.tolerance, self.pending, self.streak, self.completed_episodes = tolerance, pending, streak, count


def anneal_success_tolerance(
    env, env_ids, start_std=0.2, final_std=0.1, factor=0.9, n_succ=3, rate_threshold=0.5, window_episodes=1024
):
    """Read direct per-env sticky flags BEFORE reward reset, never extras/log.

    Three successful non-overlapping 1024-episode windows are required by
    default. This is a DexPBT-inspired tolerance curriculum, not the paper's
    consecutive manipulation count (the native task uses episode sticky flags).
    """
    state = getattr(env, "_dexpbt_anneal", None)
    if state is None:
        state = SuccessToleranceCurriculum(start_std, final_std, factor, window_episodes, n_succ, rate_threshold)
        env._dexpbt_anneal = state
    if getattr(env, "_dexpbt_fixed_adr", None) is not None:
        return {"tolerance": FIXED_POSITION_TOLERANCE, "completed_episodes": state.completed_episodes}
    term = env.reward_manager.get_term_cfg("success")
    if getattr(env, "_dexpbt_in_step", False):
        flags = term.func.succeeded[env_ids]
        eligible = getattr(env, "_dexpbt_episode_eligible", None)
        if eligible is not None:
            flags = flags[eligible[env_ids]]
        state.observe(flags.detach().cpu().tolist())
    term.params["pos_std"] = state.tolerance
    env.reward_manager.set_term_cfg("success", term)
    return {"tolerance": state.tolerance, "completed_episodes": state.completed_episodes}


def metric_condition(adr_fraction, curriculum_tolerance, **actual_parameters):
    """Exact condition stamp; never round different training difficulties together."""
    return {
        "adr_fraction": float(adr_fraction),
        "curriculum_tolerance": float(curriculum_tolerance),
        "position_tolerance": FIXED_POSITION_TOLERANCE,
        "rotation_tolerance": FIXED_ROTATION_TOLERANCE,
        "contact_threshold": FIXED_CONTACT_THRESHOLD,
        **actual_parameters,
    }


def comparison_condition(condition):
    """Shared accumulator/PBT identity based on applied ADR, not scheduler bookkeeping.

    Native ADR leaves physical parameters unchanged below its update threshold.
    A nonempty actual-parameter mapping therefore supersedes the informational
    scheduler fraction. Missing/empty/malformed stamps keep the fraction, failing
    closed rather than assuming unknown physical settings are equivalent.
    """
    result = dict(condition)
    actual = result.get("adr_parameters")
    if isinstance(actual, str):
        try:
            actual = json.loads(actual)
        except (ValueError, TypeError):
            actual = None
    if isinstance(actual, dict) and actual:
        result.pop("adr_fraction", None)
    return result


STAT_KEYS = (
    "completed_episodes",
    "success_count",
    "position_error_sum",
    "rotation_error_sum",
    "good_contact_fraction_sum",
    "episode_length_sum",
    "adr_difficulty_sum",
)


def empty_episode_report(protocol="training_conditions_fixed_definition_v1"):
    return {
        "schema_version": 1,
        "protocol": protocol,
        **dict.fromkeys(STAT_KEYS, 0),
        "mixed_condition_episodes": 0,
        "groups": [],
    }


class CompletedEpisodeMetrics:
    """Vectorized fixed-definition counters; only completed episodes cross the infos seam.

    Error/contact sums are sums of per-episode timestep means, not terminal
    errors or averages of reset-batch averages. Force columns are thumb first.
    Mixed-condition episodes remain in totals but never enter comparable groups.
    Reset/restore discards partial trajectories (physics is not checkpointed).
    """

    def __init__(self, num_envs, device):
        import torch

        self.torch = torch
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.succeeded = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.sums = torch.zeros((num_envs, 4), dtype=torch.float64, device=device)
        self.condition_ids = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.mixed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.conditions = []

    def observe(self, position_error, rotation_error, contact_forces, condition):
        """Observe the pre-reset/pre-command-resampling physical state once per control step."""
        torch = self.torch
        identity = comparison_condition(condition)
        if identity not in self.conditions:
            self.conditions.append(identity)
        index = self.conditions.index(identity)
        started = self.steps > 0
        self.mixed |= started & (self.condition_ids != index)
        self.condition_ids[~started] = index
        good = (contact_forces[:, 0] > FIXED_CONTACT_THRESHOLD) & (contact_forces[:, 1:] > FIXED_CONTACT_THRESHOLD).any(
            dim=1
        )
        finite = torch.isfinite(position_error) & torch.isfinite(rotation_error)
        # Invalid physical state must not silently become a good ranking signal.
        self.mixed |= ~finite | ~torch.isfinite(contact_forces).all(dim=1)
        self.succeeded |= (
            good & finite & (position_error < FIXED_POSITION_TOLERANCE) & (rotation_error < FIXED_ROTATION_TOLERANCE)
        )
        self.steps += 1
        self.sums[:, 0] += position_error
        self.sums[:, 1] += rotation_error
        self.sums[:, 2] += good
        self.sums[:, 3] += condition["adr_fraction"]

    def discard(self, env_ids=None):
        """Discard partial episodes without publishing them as completions."""
        ids = slice(None) if env_ids is None else env_ids
        self.steps[ids] = 0
        self.succeeded[ids] = False
        self.sums[ids] = 0
        self.condition_ids[ids] = -1
        self.mixed[ids] = False

    def complete(self, env_ids, protocol="training_conditions_fixed_definition_v1"):
        """Drain finished IDs once, preserving exact condition groups and sufficient stats."""
        torch = self.torch
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.steps.device)
        ids = ids[self.steps[ids] > 0]
        report = empty_episode_report(protocol)
        if len(ids) == 0:
            return report
        steps = self.steps[ids].cpu().tolist()
        successes = self.succeeded[ids].cpu().tolist()
        means = (self.sums[ids] / self.steps[ids, None]).cpu().tolist()
        conditions = self.condition_ids[ids].cpu().tolist()
        mixed = self.mixed[ids].cpu().tolist()
        groups = {}
        for length, success, values, condition_id, is_mixed in zip(steps, successes, means, conditions, mixed):
            row = dict(zip(STAT_KEYS, (1, int(success), values[0], values[1], values[2], length, values[3])))
            for key, value in row.items():
                report[key] += value
            if is_mixed:
                report["mixed_condition_episodes"] += 1
            else:
                group = groups.setdefault(
                    condition_id, {"condition": self.conditions[condition_id], **dict.fromkeys(STAT_KEYS, 0)}
                )
                for key, value in row.items():
                    group[key] += value
        report["groups"] = list(groups.values())
        self.discard(ids)
        return report
