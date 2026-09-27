"""Count TD-MPC2 gradient-update credit in individual environment transitions."""

import math


class UpdateSchedule:
    def __init__(self, seed_steps, updates_per_transition):
        if seed_steps < 0 or not math.isfinite(updates_per_transition) or updates_per_transition <= 0:
            raise ValueError("Require nonnegative seed steps and positive finite update density")
        self.seed_steps = int(seed_steps)
        self.updates_per_transition = float(updates_per_transition)
        self.frames = 0
        self.credit = 0.0

    def collect(self, transitions, ready):
        if not isinstance(transitions, int) or transitions <= 0:
            raise ValueError("transitions must be a positive integer")
        self.frames += transitions
        self.credit += transitions * self.updates_per_transition
        if self.frames < self.seed_steps or not ready:
            return 0
        updates = math.floor(self.credit + 1e-10)
        self.credit = max(0.0, self.credit - updates)
        return updates

    def state_dict(self):
        return {
            "seed_steps": self.seed_steps,
            "updates_per_transition": self.updates_per_transition,
            "frames": self.frames,
            "credit": self.credit,
        }

    def load_state_dict(self, state):
        if (state["seed_steps"], state["updates_per_transition"]) != (
            self.seed_steps,
            self.updates_per_transition,
        ):
            raise ValueError("Update schedule configuration differs from checkpoint")
        if state["frames"] < 0 or state["credit"] < 0 or not math.isfinite(state["credit"]):
            raise ValueError("Invalid update schedule checkpoint")
        self.frames = int(state["frames"])
        self.credit = float(state["credit"])
