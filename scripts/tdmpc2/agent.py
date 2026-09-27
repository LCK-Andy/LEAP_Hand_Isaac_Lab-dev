"""GPU-native, asynchronous-vector adapter around the official TD-MPC2 learner.

The learner and model are vendored unchanged except import/device plumbing. Planning
batches the official MPPI operations, with per-environment warm starts and chunking.
"""

import numpy as np
import random
import torch
from copy import deepcopy
from dataclasses import asdict

from .config import TDMPC2Config
from .vendor.common import math
from .vendor.tdmpc2 import TDMPC2


class Agent:
    def __init__(self, cfg: TDMPC2Config, device="cuda:0"):
        self.cfg = deepcopy(cfg)
        self.cfg.device = str(torch.device(device))
        self.device = torch.device(device)
        # The core increments the heuristic iteration count. Keep the public base cfg immutable.
        self.core = TDMPC2(deepcopy(self.cfg))
        self.model = self.core.model
        self.prev_mean = torch.zeros(cfg.num_envs, cfg.horizon, cfg.action_dim, device=self.device)
        self.update_count = 0

    @torch.no_grad()
    def act(self, obs, reset_mask, eval_mode=False):
        obs = obs.to(self.device)
        reset_mask = torch.as_tensor(reset_mask, dtype=torch.bool, device=self.device)
        if obs.shape != (self.cfg.num_envs, self.cfg.obs_dim) or reset_mask.shape != (self.cfg.num_envs,):
            raise ValueError("Expected obs[num_envs,obs_dim] and reset_mask[num_envs]")
        self.prev_mean[reset_mask] = 0
        outputs = []
        for start in range(0, self.cfg.num_envs, self.cfg.plan_chunk_size):
            end = min(start + self.cfg.plan_chunk_size, self.cfg.num_envs)
            outputs.append(self._plan_chunk(obs[start:end], self.prev_mean[start:end], eval_mode))
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def _plan_chunk(self, obs, previous, eval_mode):
        cfg, model = self.core.cfg, self.model
        n = obs.shape[0]
        z = model.encode(obs, None)
        actions = torch.empty(n, cfg.horizon, cfg.num_samples, cfg.action_dim, device=self.device)
        if cfg.num_pi_trajs > 0:
            policy_z = z[:, None].repeat(1, cfg.num_pi_trajs, 1).reshape(n * cfg.num_pi_trajs, -1)
            for t in range(cfg.horizon):
                action, _ = model.pi(policy_z, None)
                actions[:, t, : cfg.num_pi_trajs] = action.view(n, cfg.num_pi_trajs, cfg.action_dim)
                if t < cfg.horizon - 1:
                    policy_z = model.next(policy_z, action, None)
        z = z[:, None].repeat(1, cfg.num_samples, 1)
        mean = torch.zeros_like(previous)
        mean[:, :-1] = previous[:, 1:]
        std = torch.full_like(mean, cfg.max_std)
        for _ in range(cfg.iterations):
            noise = torch.randn(n, cfg.horizon, cfg.num_samples - cfg.num_pi_trajs, cfg.action_dim, device=self.device)
            actions[:, :, cfg.num_pi_trajs :] = (mean[:, :, None] + std[:, :, None] * noise).clamp(-1, 1)
            value = self._estimate_value(z, actions).nan_to_num(0)
            elite_indices = torch.topk(value.squeeze(-1), cfg.num_elites, dim=1).indices
            elite_value = value.gather(1, elite_indices[..., None])
            elite_actions = actions.gather(
                2, elite_indices[:, None, :, None].expand(n, cfg.horizon, cfg.num_elites, cfg.action_dim)
            )
            score = torch.exp(cfg.temperature * (elite_value - elite_value.max(1, keepdim=True).values))
            score = score / score.sum(1, keepdim=True)
            weights = score[:, None]
            mean = (weights * elite_actions).sum(2) / (weights.sum(2) + 1e-9)
            std = (
                ((weights * (elite_actions - mean[:, :, None]).square()).sum(2) / (weights.sum(2) + 1e-9))
                .sqrt()
                .clamp(cfg.min_std, cfg.max_std)
            )
        # Main's probability API, not vectorized_env's accidental log(log(score)).
        selected = math.gumbel_softmax_sample(score.squeeze(-1), dim=-1)
        chosen = elite_actions.gather(
            2, selected[:, None, None, None].expand(n, cfg.horizon, 1, cfg.action_dim)
        ).squeeze(2)
        action = chosen[:, 0]
        if not eval_mode:
            action = action + std[:, 0] * torch.randn_like(action)
        previous.copy_(mean)
        return action.clamp(-1, 1)

    @torch.no_grad()
    def _estimate_value(self, z, actions):
        cfg = self.core.cfg
        value, discount = 0, 1
        terminated = torch.zeros(*z.shape[:-1], 1, device=z.device)
        for t in range(cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[:, t], None), cfg)
            z = self.model.next(z, actions[:, t], None)
            value = value + discount * (1 - terminated) * reward
            discount *= self.core.discount
            if cfg.episodic:
                terminated = (terminated + (self.model.termination(z, None) > 0.5).float()).clamp(max=1)
        action, _ = self.model.pi(z, None)
        return value + discount * (1 - terminated) * self.model.Q(z, action, None, return_type="avg")

    def update(self, replay):
        """Consume sample() -> obs[H+1,B,O], action[H,B,A], reward/terminal[H,B,1], None."""
        metrics = self.core.update(replay)
        if self.update_count == 0 and any(p not in self.core.optim.state for p in self.model._Qs.parameters()):
            raise RuntimeError("Disconnected critic: optimizer state missing after learner update")
        self.update_count += 1
        return metrics

    @torch.no_grad()
    def critic_diagnostics(self):
        """Expose critic learning evidence without treating finite losses as sufficient."""
        parameters = list(self.model._Qs.parameters())
        head = self.model._Qs.params["2", "weight"]
        target = self.model._target_Qs_params["2", "weight"]
        return {
            "critic/parameter_tensors": len(parameters),
            "critic/optimizer_tensors": sum(p in self.core.optim.state for p in parameters),
            "critic/head_abs_mean": float(head.abs().mean()),
            "critic/target_head_abs_mean": float(target.abs().mean()),
            "critic/optimizer_steps_min": min(
                (float(self.core.optim.state[p]["step"]) for p in parameters if p in self.core.optim.state),
                default=0.0,
            ),
        }

    def set_hparams(self, values):
        """Apply bounded live PBT changes without resetting either Adam optimizer."""
        bounds = {"lr": (1e-6, 1e-2), "entropy_coef": (0.0, 0.1), "grad_clip_norm": (0.1, 1000.0)}
        for name, value in values.items():
            if name not in bounds:
                raise ValueError(f"Unsupported live hyperparameter: {name}")
            lo, hi = bounds[name]
            if not lo <= float(value) <= hi:
                raise ValueError(f"{name} must be finite and in [{lo}, {hi}]")
        for name, value in values.items():
            setattr(self.cfg, name, float(value))
            setattr(self.core.cfg, name, float(value))
        if "lr" in values:
            for index, group in enumerate(self.core.optim.param_groups):
                group["lr"] = self.cfg.lr * (self.cfg.enc_lr_scale if index == 0 else 1)
            for group in self.core.pi_optim.param_groups:
                group["lr"] = self.cfg.lr

    def state_dict(self):
        """Return a detached snapshot; includes learner/RNG, not simulator or replay state."""
        return deepcopy(
            {
                "version": 1,
                "config": asdict(self.cfg),
                "model": self.model.state_dict(),
                "optim": self.core.optim.state_dict(),
                "pi_optim": self.core.pi_optim.state_dict(),
                "scale": self.core.scale.state_dict(),
                "prev_mean": self.prev_mean,
                "update_count": self.update_count,
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            }
        )

    def load_state_dict(self, state, *, reset_planner=False, restore_rng=True):
        """Restore learner; reset_planner=True when the simulator cannot be restored.

        A PBT restart normally transfers optimizer/target/scale state but resets the
        planner. Replay, environment state and recipient frame accounting belong to
        the trainer. restore_rng=False retains recipient random streams on exploit.
        """
        if state.get("version") != 1:
            raise ValueError("Unsupported full-state TD-MPC2 checkpoint")
        allowed = {"device", "num_envs", "plan_chunk_size", "compile", "lr", "entropy_coef", "grad_clip_norm"}
        for key, value in asdict(self.cfg).items():
            if key not in allowed and state["config"].get(key) != value:
                raise ValueError(f"Checkpoint configuration mismatch for {key}")
        self.model.load_state_dict(state["model"], strict=True)
        self.core.optim.load_state_dict(deepcopy(state["optim"]))
        self.core.pi_optim.load_state_dict(deepcopy(state["pi_optim"]))
        self.core.scale.load_state_dict(state["scale"])
        self.set_hparams({key: state["config"][key] for key in ("lr", "entropy_coef", "grad_clip_norm")})
        if reset_planner:
            self.prev_mean.zero_()
        else:
            if state["prev_mean"].shape != self.prev_mean.shape:
                raise ValueError("Planner shape differs; use reset_planner=True after environment restart")
            self.prev_mean.copy_(state["prev_mean"])
        self.update_count = int(state["update_count"])
        if restore_rng:
            random.setstate(state["rng"]["python"])
            np.random.set_state(state["rng"]["numpy"])
            torch.set_rng_state(state["rng"]["torch"].cpu())
            if state["rng"]["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([value.cpu() for value in state["rng"]["cuda"]])
        self.model.eval()
