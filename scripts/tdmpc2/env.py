"""Tensor-only TD-MPC2 boundary for the preserved Dexsuite KUKA/Allegro task.

Import-safe before AppLauncher: this module deliberately imports no Isaac packages.
"""

import torch
from dataclasses import dataclass
from typing import Any


@dataclass
class StepResult:
    """Acting observations and replay successors are distinct at same-step resets."""

    obs: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    infos: dict[str, Any]


class _HistoryPreview:
    """Append one fresh sample to a view of history without writing native buffers."""

    def __init__(self, history):
        self.history = history
        self._buffer = history._buffer
        self.buffer = None

    def append(self, value):
        if self._buffer is None:
            raise RuntimeError("Terminal observation history is uninitialized; call reset() first")
        previous = self.history.buffer
        self.buffer = torch.cat((previous[:, 1:], value.unsqueeze(1)), dim=1)


class DexsuiteVectorEnv:
    """Preserve native group/term/history order and native action-manager scaling."""

    obs_dim = 1745
    action_dim = 23
    _groups = (("policy", 170), ("proprio", 615), ("perception", 960))

    def __init__(self, env):
        self.env = env.unwrapped
        self.num_envs = self.env.num_envs
        self.device = torch.device(self.env.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.env.cfg.compute_final_obs:
            raise ValueError("Keep compute_final_obs=False; native compute(False) returns stale histories")
        if self.env.recorder_manager.active_terms:
            raise ValueError("Observation recorders are not supported by terminal preview")
        manager = self.env.observation_manager
        if manager._group_obs_class_instances:
            raise ValueError("Stateful observation modifiers/noise are not supported by terminal preview")
        self._point_cloud_terms = []
        for terms in manager._group_obs_class_term_cfgs.values():
            for term in terms:
                cls = type(term.func)
                if (
                    cls.__name__ != "object_point_cloud_b"
                    or cls.__module__ != "LEAP_Isaaclab.compat.dexsuite.mdp.observations"
                ):
                    raise ValueError(f"Unvalidated stateful observation term: {cls.__module__}.{cls.__name__}")
                self._point_cloud_terms.append(term)

    def _flatten(self, observations):
        groups = []
        for name, width in self._groups:
            value = observations[name]
            if not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, width):
                raise ValueError(f"Expected native {name} observations of shape ({self.num_envs}, {width})")
            if value.device != self.device:
                raise ValueError(f"Native {name} observations must stay on {self.device}")
            groups.append(value)
        return torch.cat(groups, dim=-1)

    def reset(self, seed=None):
        observations, _ = self.env.reset(seed=seed)
        return self._flatten(observations)

    def get_state(self):
        """Forward curriculum/ADR state, not a simulator or observation checkpoint."""
        return self.env.get_env_state()

    def set_state(self, state):
        return self.env.set_env_state(state)

    def close(self):
        self.env.close()

    def _terminal_observations(self):
        manager = self.env.observation_manager
        histories = manager._group_obs_term_history_buffer
        cached = manager._obs_buffer
        point_cloud_state = [(term, term.params, term.func.points_w) for term in self._point_cloud_terms]
        try:
            for term, params, _ in point_cloud_state:
                term.params = {**params, "visualize": False}
            manager._group_obs_term_history_buffer = {
                group: {name: _HistoryPreview(history) for name, history in terms.items()}
                for group, terms in histories.items()
            }
            # Native Dexsuite corruption draws only Torch CPU/device RNG. Preview
            # must not change reset randomization or any subsequent acting sample.
            devices = [self.device.index] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                return self._flatten(manager.compute(update_history=True))
        finally:
            manager._group_obs_term_history_buffer = histories
            manager._obs_buffer = cached
            for term, params, points_w in point_cloud_state:
                term.params = params
                term.func.points_w = points_w

    def step(self, action):
        if not isinstance(action, torch.Tensor) or action.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Expected action shape ({self.num_envs}, {self.action_dim})")
        if action.device != self.device:
            raise ValueError(f"Expected action on {self.device}; host transfers are not performed")
        captured = []
        original_reset = self.env._reset_idx
        had_override = "_reset_idx" in vars(self.env)

        def capture_before_reset(env_ids):
            captured.append((env_ids.clone().long(), self._terminal_observations()[env_ids].clone()))
            return original_reset(env_ids)

        self.env._reset_idx = capture_before_reset
        try:
            observations, reward, terminated, truncated, infos = self.env.step(action.clamp(-1, 1))
        finally:
            if had_override:
                self.env._reset_idx = original_reset
            else:
                del self.env._reset_idx
        obs = self._flatten(observations)
        next_obs = obs.clone()
        captured_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for ids, final in captured:
            next_obs[ids] = final
            captured_mask[ids] = True
        if not torch.equal(captured_mask, terminated | truncated):
            raise RuntimeError("Missing or mismatched pre-reset terminal observation capture")
        return StepResult(obs, next_obs, reward, terminated, truncated, infos)
