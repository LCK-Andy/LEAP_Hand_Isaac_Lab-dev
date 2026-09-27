"""Import-safe native Stage A boundary, sharing only finite-horizon plumbing with Stage B.

The tasks own forces, actions, reward and persistent success. No motion commands,
force reconstruction, reward shaping, or extra action integration live here.
"""

import math
import torch
from collections import deque
from copy import deepcopy

from .env import StepResult


def _matches(value, expected):
    if isinstance(expected, tuple):
        return (
            isinstance(value, (tuple, list))
            and len(value) == len(expected)
            and all(_matches(a, b) for a, b in zip(value, expected))
        )
    if isinstance(expected, bool):
        return value is expected
    return isinstance(value, (int, float)) and math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12)


def _solver_properties(cfg):
    properties = cfg.robot_cfg.spawn.articulation_props
    if isinstance(properties, (tuple, list)):
        if len(properties) != 1:
            raise ValueError("Grasp requires one native hand articulation configuration")
        properties = properties[0]
    return properties


def _validate_grasp_cfg(cfg, obs_dim=124, stage="Stage A"):
    expected = {
        "observation_space": obs_dim,
        "action_space": 16,
        "episode_length_s": 10.0,
        "is_finite_horizon": True,
        "decimation": 4,
        "max_joint_delta": 0.015,
        "object_mass": 0.216,
        "cube_half_extent": 0.036,
        "hold_duration": 2.0,
        "fingertip_contact_sensors": True,
        "allow_unvalidated_backend": False,
    }
    for name, value in expected.items():
        if not _matches(getattr(cfg, name, None), value):
            raise ValueError(f"{stage} requires native {name}={value}")
    if not _matches(cfg.sim.dt, 1 / 120) or cfg.sim.use_newton_actuators:
        raise ValueError(f"{stage} requires native 120 Hz physics with per-substep contact history")
    if type(cfg.sim.physics).__name__ != "PhysxCfg":
        raise ValueError(f"{stage} TD-MPC2 training requires the validated PhysX backend")
    properties = _solver_properties(cfg)
    if properties.solver_position_iteration_count != 8 or properties.solver_velocity_iteration_count != 0:
        raise ValueError(f"{stage} requires native hand solver iterations 8/0")
    if cfg.observation_noise_model is not None or cfg.action_noise_model is not None:
        raise ValueError(f"{stage} additional action/observation noise is not validated")


def configure_grasp(cfg, seed):
    """Keep native 124-channel full-force observations and 2s hold; enable terminal capture."""
    _validate_grasp_cfg(cfg)
    cfg.seed = seed
    cfg.compute_final_obs = True
    return cfg


class _FiniteHorizonGraspVectorEnv:
    """Shared EA final-observation capture; subclasses supply only their native success field."""

    action_dim = 16

    def __init__(self, env):
        self.env = env.unwrapped
        self._validate_cfg(self.env.cfg)
        if not self.env.cfg.compute_final_obs:
            raise ValueError(f"{self._stage} requires native compute_final_obs=True")
        if (
            getattr(self.env, "grasp_hold_contract_version", None) != 1
            or self.env.max_episode_length != 300
            or not _matches(self.env.step_dt, 1 / 30)
            or getattr(self.env, "_physics_backend", None) != "physx"
        ):
            raise ValueError(f"{self._stage} native runtime contract changed")
        self.num_envs = self.env.num_envs
        self.device = torch.device(self.env.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

    def condition(self):
        """Use the task's physical stamp, never a trainer-invented condition."""
        key = self.env.condition_key
        if not isinstance(key, str) or not key.startswith(self._condition_prefix):
            raise ValueError(f"{self._stage} physical condition stamp is missing")
        return key

    def contract_metadata(self):
        return {
            "observation_groups": ["policy"],
            "episode_length": 300,
            "finite_horizon_terminal": True,
            "physics_dt": self.env.cfg.sim.dt,
            "decimation": self.env.cfg.decimation,
            "solver_position_iterations": _solver_properties(self.env.cfg).solver_position_iteration_count,
            "solver_velocity_iterations": _solver_properties(self.env.cfg).solver_velocity_iteration_count,
            "max_joint_delta": self.env.cfg.max_joint_delta,
            "condition_key": self.condition(),
        }

    def get_state(self):
        state = self.env.get_env_state()
        if state is not None:
            raise ValueError(f"{self._stage} has no checkpointed environment state")
        return None

    def set_state(self, state):
        if state is not None:
            raise ValueError(f"{self._stage} cannot restore nonempty environment state")
        self.env.set_env_state(None)

    def close(self):
        self.env.close()

    def _observations(self, observations):
        value = observations.get("policy") if isinstance(observations, dict) else None
        if not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, self.obs_dim):
            raise ValueError(f"Expected full policy observations ({self.num_envs}, {self.obs_dim})")
        if value.device != self.device:
            raise ValueError(f"Observations must remain on {self.device}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite {self._stage} observations")
        return value

    def _vector(self, value, name, boolean=False):
        if not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,):
            raise ValueError(f"Expected native {name} vector shape ({self.num_envs},)")
        if value.device != self.device:
            raise ValueError(f"Native {name} must remain on {self.device}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite native {name}")
        if boolean and value.dtype != torch.bool:
            raise ValueError(f"Native {name} must be boolean")
        return value

    def reset(self, seed=None):
        observations, _ = self.env.reset(seed=seed)
        return self._observations(observations).clone()

    def step(self, action):
        if not isinstance(action, torch.Tensor) or action.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Expected action ({self.num_envs}, {self.action_dim})")
        if action.device != self.device or not torch.isfinite(action).all():
            raise ValueError(f"Actions must be finite and remain on {self.device}")
        # EA retains extras across steps, including final_obs on a nonterminal step.
        for key in ("final_obs", "_final_obs", *self._report_groups):
            self.env.extras.pop(key, None)
        captured = []
        original_reset = self.env._reset_idx
        had_override = "_reset_idx" in vars(self.env)

        def capture_before_reset(env_ids):
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if not (self.env.grasp_state.steps[ids] > 0).all():
                raise RuntimeError(f"Unexpected non-episode reset during {self._stage} step")
            if "final_obs" not in self.env.extras:
                raise RuntimeError("Missing fresh pre-reset terminal observations")
            final = self._observations(self.env.extras["final_obs"])[ids].clone()
            success = self._vector(self._native_success(), "success", boolean=True)[ids].clone()
            captured.append((ids.clone(), final, success))
            return original_reset(env_ids)

        self.env._reset_idx = capture_before_reset
        try:
            observations, reward, terminated, truncated, infos = self.env.step(action.clamp(-1, 1))
        finally:
            if had_override:
                self.env._reset_idx = original_reset
            else:
                del self.env._reset_idx
        reward = self._vector(reward, "reward")
        terminated = self._vector(terminated, "terminated", boolean=True).clone()
        truncated = self._vector(truncated, "truncated", boolean=True).clone()
        done = terminated | truncated
        obs = self._observations(observations).clone()
        next_obs = obs.clone()
        captured_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for ids, final, _ in captured:
            if captured_mask[ids].any():
                raise RuntimeError("Duplicate terminal reset capture")
            captured_mask[ids] = True
            next_obs[ids] = final
        if not torch.equal(captured_mask, done):
            raise RuntimeError("Missing or mismatched terminal observation capture")
        mask = infos.get("_final_obs")
        if mask is not None and (not isinstance(mask, torch.Tensor) or not torch.equal(mask, done)):
            raise RuntimeError("Incomplete or mismatched final observation mask")
        infos = dict(infos)
        infos[self._info_key] = {
            "native_terminated": terminated,
            "native_truncated": truncated.clone(),
        }
        if captured:
            infos[self._report_group] = dict(infos.get(self._report_group, {}))
            infos[self._report_group][self._success_key] = torch.cat([entry[2] for entry in captured])
        else:
            for group in self._report_groups:
                infos[group] = {}
        # A 10-second task is genuinely finite-horizon, not a continuing task's time-limit cut.
        return StepResult(obs, next_obs, reward.clone(), done, truncated, infos)


class GraspVectorEnv(_FiniteHorizonGraspVectorEnv):
    """Native Stage A: one bounded 16-action update, full 124-state observations, no PBT."""

    obs_dim = 124
    _stage = "Stage A"
    _condition_prefix = "grasp_hold_v1_"
    _report_group = "grasp_hold"
    _report_groups = ("grasp_hold",)
    _success_key = "completed_grasp_success"
    _info_key = "tdmpc2_grasp"
    _validate_cfg = staticmethod(_validate_grasp_cfg)

    def _native_success(self):
        return self.env.grasp_state.success


class _GraspEpisodeMetrics:
    """Common completed-episode accounting, with subclass-owned native report semantics."""

    def __init__(self, num_envs, device, recent_window=100):
        if type(recent_window) is not int or recent_window < 1:
            raise ValueError("recent_window must be a positive integer")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.recent_window = recent_window
        self.returns = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self.lengths = torch.zeros_like(self.returns)
        self.totals = {"completed": 0, "success_count": 0, **{f"{name}_sum": 0.0 for name in self._fields}}
        self.recent = deque(maxlen=recent_window)
        self.condition_key = None

    def observe(self, reward, done, infos):
        reward = torch.as_tensor(reward, device=self.device)
        done = torch.as_tensor(done, device=self.device)
        if reward.shape != (self.num_envs,) or done.shape != (self.num_envs,) or done.dtype != torch.bool:
            raise ValueError("Expected vector reward and boolean done arrays")
        if not torch.isfinite(reward).all():
            raise FloatingPointError(f"Nonfinite {self._stage} episode reward")
        ids = done.nonzero().flatten()
        if not len(ids):
            self.returns += reward
            self.lengths += 1
            return  # Persistent extras/log dictionaries must not be read without an ending.
        columns, condition = self._report(infos, len(ids))
        # Validate complete native reports before mutating any accounting state.
        self.returns += reward
        self.lengths += 1
        self.condition_key = condition
        returns = self.returns[ids].tolist()
        lengths = self.lengths[ids].tolist()
        for index in range(len(ids)):
            row = {name: values[index] for name, values in columns.items()}
            row.update({"return": returns[index], "length": lengths[index]})
            self._derive(row)
            row["success"] = int(row["success"])
            self.totals["completed"] += 1
            self.totals["success_count"] += row["success"]
            for name in self._fields:
                self.totals[f"{name}_sum"] += row[name]
            self.recent.append(row)
        self.returns[done] = 0
        self.lengths[done] = 0

    def state_dict(self):
        """Persist complete-episode statistics only; resume starts new physical episodes."""
        return deepcopy(
            {
                "version": 1,
                "recent_window": self.recent_window,
                "condition_key": self.condition_key,
                "totals": self.totals,
                "recent": list(self.recent),
            }
        )

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("recent_window") != self.recent_window:
            raise ValueError(f"Incompatible {self._stage} metrics checkpoint")
        totals, recent = state.get("totals", {}), state.get("recent", [])
        if totals.keys() != self.totals.keys() or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) for value in totals.values()
        ):
            raise ValueError(f"Invalid {self._stage} metrics checkpoint totals")
        count, successes = totals["completed"], totals["success_count"]
        if type(count) is not int or type(successes) is not int or not 0 <= successes <= count:
            raise ValueError(f"Invalid {self._stage} metrics checkpoint counts")
        if len(recent) != min(count, self.recent_window):
            raise ValueError(f"Incomplete {self._stage} metrics checkpoint recent window")
        for name in self._fields:
            if name not in {"return", "length"} and not 0 <= totals[f"{name}_sum"] <= count:
                raise ValueError(f"Invalid {self._stage} metrics checkpoint fraction sum")
        if not count <= totals["length_sum"] <= 300 * count:
            raise ValueError(f"Invalid {self._stage} metrics checkpoint episode lengths")
        for row in recent:
            if row.keys() != {*self._fields, "success"} or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) for value in row.values()
            ):
                raise ValueError(f"Invalid {self._stage} metrics checkpoint recent record")
            if row["success"] not in (0, 1) or not 1 <= row["length"] <= 300:
                raise ValueError(f"Invalid {self._stage} metrics checkpoint recent success/length")
            if any(not 0 <= row[name] <= 1 for name in self._fields if name not in {"return", "length"}):
                raise ValueError(f"Invalid {self._stage} metrics checkpoint recent fraction")
        condition = state.get("condition_key")
        if count and (not isinstance(condition, str) or not condition.startswith(self._condition_prefix)):
            raise ValueError(f"Invalid {self._stage} metrics checkpoint physical condition")
        self.totals = deepcopy(totals)
        self.recent = deque(deepcopy(recent), maxlen=self.recent_window)
        self.condition_key = condition
        self.returns.zero_()
        self.lengths.zero_()

    def values(self):
        count = self.totals["completed"]
        values = {
            "episode/completed": count,
            "episode/measured": count,
            "episode/success_count": self.totals["success_count"],
        }
        if count:
            values["episode/success_rate"] = self.totals["success_count"] / count
            for name in self._fields:
                prefix = "episode" if name in {"return", "length"} else self._metric_prefix
                values[f"{prefix}/{name}"] = self.totals[f"{name}_sum"] / count
        if self.recent:
            values["recent/completed"] = len(self.recent)
            values["recent/success_rate"] = sum(row["success"] for row in self.recent) / len(self.recent)
            for name in self._fields:
                values[f"recent/{name}"] = sum(row[name] for row in self.recent) / len(self.recent)
        return values


class GraspEpisodeMetrics(_GraspEpisodeMetrics):
    """Native persistent 2s success, separately from support-free retention and valid occupancy.

    ``safe_success`` means the native hold succeeded AND the episode finished with
    support-free retention. It does not redefine the task's sticky success after a drop.
    """

    _stage = "Stage A"
    _condition_prefix = "grasp_hold_v1_"
    _metric_prefix = "grasp_hold"
    _fields = ("return", "length", "retained", "valid_fraction", "overload_fraction", "safe_success")

    def _derive(self, row):
        row["safe_success"] = row["success"] * row["retained"]

    def _report(self, infos, count):
        grasp = infos.get("grasp_hold", {})
        condition = grasp.get("condition_key")
        if (
            not isinstance(condition, str)
            or not condition.startswith(self._condition_prefix)
            or self.condition_key not in (None, condition)
        ):
            raise ValueError("Native Stage A report physical conditions differ or are missing")
        columns = {}
        for name, key in (
            ("retained", "completed_retained"),
            ("valid_fraction", "completed_valid_fraction"),
            ("overload_fraction", "completed_overload_fraction"),
            ("success", "completed_grasp_success"),
        ):
            value = grasp.get(key)
            if not isinstance(value, torch.Tensor) or value.shape != (count,):
                raise ValueError(f"Native Stage A report {key} must have exactly {count} completed entries")
            value = value.detach().to(device="cpu", dtype=torch.float64)
            if not torch.isfinite(value).all() or not ((value >= 0) & (value <= 1)).all():
                raise ValueError(f"Native Stage A report {key} must be finite and within [0, 1]")
            if name in {"retained", "success"} and not ((value == 0) | (value == 1)).all():
                raise ValueError(f"Native Stage A report {key} must be boolean")
            columns[name] = value.tolist()
        return columns, condition
