"""Thin native-task adapter for pre-reset measurements and rl-games env state.

The native step/reset/reward implementation is retained. A reward-manager compute
adapter measures the same physical state immediately AFTER native reward compute,
BEFORE resets and command resampling. No reward value or weight is changed here.
"""

import copy
import json
import math
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

from .evaluation import (
    FIXED_CONTACT_THRESHOLD,
    FIXED_POSITION_TOLERANCE,
    FIXED_ROTATION_TOLERANCE,
    CompletedEpisodeMetrics,
    SuccessToleranceCurriculum,
    empty_episode_report,
    metric_condition,
)


class DexsuiteReorientPbtEnv(ManagerBasedRLEnv):
    """Native DexSuite environment with completed-episode sufficient statistics."""

    def load_managers(self):
        super().load_managers()
        self._dexpbt_fixed_adr = None
        self._dexpbt_in_step = False
        self._dexpbt_metrics = CompletedEpisodeMetrics(self.num_envs, self.device)
        self._dexpbt_episode_eligible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        anneal = getattr(self.curriculum_manager.cfg, "anneal_success_tolerance", None)
        self._dexpbt_anneal = None
        if anneal is not None:
            params = dict(anneal.params)
            params["required_windows"] = params.pop("n_succ")
            self._dexpbt_anneal = SuccessToleranceCurriculum(**params)
            term = self.reward_manager.get_term_cfg("success")
            term.params["pos_std"] = self._dexpbt_anneal.tolerance
            self.reward_manager.set_term_cfg("success", term)
        # Only identifiers are taken from native config. Thresholds never follow
        # reward mutation, native sticky flags, or the annealed training tolerance.
        params = self.reward_manager.get_term_cfg("success").params
        self._dexpbt_measurement = {
            key: copy.deepcopy(params[key])
            for key in ("command_name", "asset_cfg", "align_asset_cfg", "thumb_name", "finger_names")
        }
        native_compute = self.reward_manager.compute

        def compute_and_measure(dt):
            reward = native_compute(dt=dt)
            if self._dexpbt_in_step:
                self._measure_episode_step()
            return reward

        self.reward_manager.compute = compute_and_measure

    @property
    def evaluation_protocol(self):
        prefix = "heldout" if self._dexpbt_fixed_adr is not None else "training_conditions"
        return f"{prefix}_fixed_definition_v1"

    def step(self, action):
        self.extras["dexpbt"] = empty_episode_report(self.evaluation_protocol)
        self._dexpbt_in_step = True
        try:
            return super().step(action)
        finally:
            self._dexpbt_in_step = False

    def _reset_idx(self, env_ids):
        # Native step supplies int32 indices; the reset path repeatedly indexes Torch
        # buffers. Normalize once instead of re-casting at each advanced-index operation.
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if hasattr(self, "_dexpbt_metrics"):
            if self._dexpbt_in_step:
                report = self._dexpbt_metrics.complete(
                    env_ids[self._dexpbt_episode_eligible[env_ids]], self.evaluation_protocol
                )
                self._dexpbt_metrics.discard(env_ids)
                report["curriculum_tolerance"] = self.reward_manager.get_term_cfg("success").params["pos_std"]
                self.extras["dexpbt"] = report
            else:
                self._dexpbt_metrics.discard(env_ids)
                self.extras["dexpbt"] = empty_episode_report(self.evaluation_protocol)
        super()._reset_idx(env_ids)
        if hasattr(self, "_dexpbt_episode_eligible"):
            self._dexpbt_episode_eligible[env_ids] = True

    @staticmethod
    def _tensor(value):
        return value.torch if hasattr(value, "torch") else value

    def _measure_episode_step(self):
        params = self._dexpbt_measurement
        base = self.scene[params["asset_cfg"].name].data
        obj = self.scene[params["align_asset_cfg"].name].data
        command = self.command_manager.get_command(params["command_name"])
        desired_pos, desired_quat = combine_frame_transforms(
            self._tensor(base.root_pos_w), self._tensor(base.root_quat_w), command[:, :3], command[:, 3:7]
        )
        position, rotation = compute_pose_error(
            desired_pos, desired_quat, self._tensor(obj.root_pos_w), self._tensor(obj.root_quat_w)
        )
        forces = []
        for name in [params["thumb_name"], *params["finger_names"]]:
            force = self._tensor(self.scene.sensors[name].data.force_matrix_w)
            # Same physical filtered sensor reduction as the native contacts helper.
            forces.append(force.reshape(self.num_envs, 3).norm(dim=-1))
        self._dexpbt_metrics.observe(
            position.norm(dim=1), rotation.norm(dim=1), torch.stack(forces, dim=1), self.get_evaluation_condition()
        )

    def _adr_values(self):
        """Read actual mutable observation-noise/gravity parameters, including ADR hysteresis."""
        result = {}
        for name in self.curriculum_manager.active_terms:
            cfg = getattr(self.curriculum_manager.cfg, name)
            if "modify_params" not in cfg.params or cfg.params["modify_params"].get("difficulty_term_str") != "adr":
                continue
            address = cfg.params["address"].replace("s.", "_manager.cfg.", 1)
            value = self
            for part in address.split("."):
                value = value[part] if isinstance(value, dict) else getattr(value, part)
            result[name] = copy.deepcopy(value)
        return result

    def get_evaluation_condition(self):
        """Return exact comparison stamp; online ADR is never described as heldout eval."""
        adr = self.curriculum_manager.cfg.adr.func
        return metric_condition(
            float(adr.difficulty_frac),
            self.reward_manager.get_term_cfg("success").params["pos_std"],
            adr_parameters=json.dumps(self._adr_values(), sort_keys=True),
            protocol=self.evaluation_protocol,
        )

    def _apply_adr_values(self, values):
        for name, value in values.items():
            cfg = getattr(self.curriculum_manager.cfg, name)
            # Public native term API performs the same manager-config write as ADR,
            # but without the native <.1 NO_CHANGE hysteresis on restoration/freeze.
            cfg.func(self, [], **{**cfg.params, "modify_fn": lambda *args, value=value, **kwargs: value})

    def configure_fixed_evaluation(self, adr_fraction=0.0):
        """Freeze shared difficulty; call AFTER checkpoint restore, BEFORE seeded reset.

        Caller owns heldout seeds, policy mode, episode budget and output manifest.
        This does not alter startup randomization or create a heldout rollout itself.
        """
        if not math.isfinite(adr_fraction) or not 0 <= adr_fraction <= 1:
            raise ValueError("adr_fraction must lie in [0, 1]")
        self._dexpbt_fixed_adr = float(adr_fraction)
        adr_cfg = self.curriculum_manager.cfg.adr
        adr_cfg.func.current_adr_difficulties.fill_(adr_fraction * adr_cfg.params["max_difficulty"])
        adr_cfg.func.difficulty_frac = torch.tensor(adr_fraction, device=self.device)
        values = {}

        def interpolate(initial, final):
            if isinstance(initial, (list, tuple)):
                return type(initial)(interpolate(a, b) for a, b in zip(initial, final))
            return initial + adr_fraction * (final - initial)

        for name in self._adr_values():
            params = getattr(self.curriculum_manager.cfg, name).params["modify_params"]
            values[name] = interpolate(params["initial_value"], params["final_value"])
        self._apply_adr_values(values)
        term = self.reward_manager.get_term_cfg("success")
        term.params.update(
            pos_std=FIXED_POSITION_TOLERANCE,
            rot_std=FIXED_ROTATION_TOLERANCE,
            contact_threshold=FIXED_CONTACT_THRESHOLD,
        )
        self.reward_manager.set_term_cfg("success", term)
        self._dexpbt_metrics.discard()
        # A mid-episode restore/freeze cannot turn the remaining suffix into a
        # completed episode. Eligibility resumes only after a real reset.
        self._dexpbt_episode_eligible[:] = False
        self.extras["dexpbt"] = empty_episode_report(self.evaluation_protocol)

    def get_env_state(self):
        """JSON-safe curriculum state for rl_games checkpoints, not simulator/episode state."""
        return {
            "dexpbt": {
                "version": 1,
                "curriculum": self._dexpbt_anneal.get_state() if self._dexpbt_anneal is not None else None,
                "adr_difficulties": self.curriculum_manager.cfg.adr.func.current_adr_difficulties.cpu().tolist(),
                "adr_values": self._adr_values(),
                "fixed_adr": self._dexpbt_fixed_adr,
            }
        }

    def set_env_state(self, env_state):
        """Restore donor curriculum; discard recipient measurement window/partial episodes.

        Different vector counts cannot faithfully restore per-env ADR; reject rather
        than silently average. Heldout playback should instantiate matching counts
        or explicitly omit training env-state restoration and then freeze difficulty.
        """
        if not env_state or "dexpbt" not in env_state:
            return
        state = env_state["dexpbt"]
        if state.get("version") != 1:
            raise ValueError("Unsupported DexPBT environment state")
        difficulties = torch.tensor(state["adr_difficulties"], device=self.device)
        if difficulties.shape != (self.num_envs,) or not torch.isfinite(difficulties).all():
            raise ValueError("ADR checkpoint requires matching num_envs and finite difficulties")
        max_difficulty = self.curriculum_manager.cfg.adr.params["max_difficulty"]
        if ((difficulties < 0) | (difficulties > max_difficulty)).any():
            raise ValueError("ADR checkpoint difficulty outside configured range")
        if state["curriculum"] is not None and self._dexpbt_anneal is not None:
            self._dexpbt_anneal.set_state(state["curriculum"])
            term = self.reward_manager.get_term_cfg("success")
            term.params["pos_std"] = self._dexpbt_anneal.tolerance
            self.reward_manager.set_term_cfg("success", term)
        adr = self.curriculum_manager.cfg.adr.func
        adr.set_state(difficulties)
        adr.difficulty_frac = difficulties.mean() / max(max_difficulty, 1)
        self._apply_adr_values(state["adr_values"])
        self._dexpbt_fixed_adr = state.get("fixed_adr")
        if self._dexpbt_fixed_adr is not None:
            self.configure_fixed_evaluation(self._dexpbt_fixed_adr)
        self._dexpbt_metrics.discard()
        # A mid-episode restore/freeze cannot turn the remaining suffix into a
        # completed episode. Eligibility resumes only after a real reset.
        self._dexpbt_episode_eligible[:] = False
        self.extras["dexpbt"] = empty_episode_report(self.evaluation_protocol)
