# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# DP3-style point-set model for rsl_rl (used by Isaac-Repose-Cube-LEAP-Direct-v2):
# a shared per-point MLP followed by masked max-pooling per point set
# (object / hand), concatenated with the normalized 1D observation groups.
# --------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.utils.utils import resolve_callable


class LeapPointSetModel(MLPModel):
    """MLPModel with a DP3-style point-set encoder for one extra obs group.

    The point group must have shape (N, P, C) with the first ``point_split``
    rows being object points and the rest hand points; all-zero rows are unused
    slots. Each point goes through a shared MLP; the two sets are pooled with
    masked max-pooling and concatenated with the (normalized) 1D groups to form
    the latent for the MLP head. Constructor signature mirrors ``MLPModel`` so
    it can be built by ``PPO.construct_algorithm`` via ``class_name``.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        point_group: str = "points",
        point_split: int = 128,
        point_hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        point_latent_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        # note: deliberately NOT calling MLPModel.__init__ (it rejects >1D groups);
        # the pieces below mirror it with the point branch added.
        nn.Module.__init__(self)

        self._point_group = point_group
        self._point_split = point_split
        self._point_latent_dim = point_latent_dim

        active_groups = obs_groups[obs_set]
        if point_group not in active_groups:
            raise ValueError(f"obs group '{point_group}' missing from obs_groups[{obs_set}]")
        self.obs_groups = [g for g in active_groups if g != point_group]

        point_shape = obs[point_group].shape
        if len(point_shape) != 3:
            raise ValueError(f"Point group '{point_group}' must be (N, P, C), got {point_shape}")
        self._point_channels = point_shape[-1]

        self.obs_dim = 0
        for group in self.obs_groups:
            if len(obs[group].shape) != 2:
                raise ValueError(f"Expected 1D obs group '{group}', got {obs[group].shape}")
            self.obs_dim += obs[group].shape[-1]

        # per-point shared MLP encoder (DP3-style), followed by per-set masked max-pool
        self.point_net = MLP(
            self._point_channels, point_latent_dim, list(point_hidden_dims), activation
        )

        # observation normalization (1D groups only)
        self.obs_normalization = obs_normalization
        if obs_normalization:
            from rsl_rl.modules.normalization import EmpiricalNormalization

            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        # distribution + MLP head, mirroring MLPModel
        if distribution_cfg is not None:
            dist_class = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim
        self.mlp = MLP(self.obs_dim + 2 * point_latent_dim, mlp_output_dim, list(hidden_dims), activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def _pool_point_set(self, encoded: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Masked max-pool of encoded[start:end] over the point axis; zeros if empty."""
        subset = encoded[:, start:end, :]  # (N, K, L)
        nonempty = torch.linalg.norm(subset, dim=-1) > 0.0  # zero rows are unused slots
        masked = torch.where(nonempty.unsqueeze(-1), subset, torch.full_like(subset, -1e9))
        pooled = masked.max(dim=1).values
        return torch.where(nonempty.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))

    def _encode_points(self, obs: TensorDict) -> torch.Tensor:
        points = obs[self._point_group]  # (N, P, C)
        encoded = self.point_net(points)  # (N, P, L)
        object_latent = self._pool_point_set(encoded, 0, self._point_split)
        hand_latent = self._pool_point_set(encoded, self._point_split, points.shape[1])
        return torch.cat((object_latent, hand_latent), dim=-1)  # (N, 2L)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
    ) -> torch.Tensor:
        obs_1d = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        return torch.cat((self.obs_normalizer(obs_1d), self._encode_points(obs)), dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_1d = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
            self.obs_normalizer.update(obs_1d)

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("JIT export is not implemented for LeapPointSetModel.")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        raise NotImplementedError("ONNX export is not implemented for LeapPointSetModel.")
