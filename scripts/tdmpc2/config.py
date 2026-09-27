"""Explicit single-task configuration for the pinned official TD-MPC2 core."""

from dataclasses import dataclass, field


@dataclass
class TDMPC2Config:
    obs_dim: int = 1745
    action_dim: int = 23
    num_envs: int = 256
    plan_chunk_size: int = 32
    episode_length: int = 360
    batch_size: int = 256
    horizon: int = 3
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    # Official base count; the core adds two exactly once for action_dim >= 20.
    iterations: int = 6
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5
    lr: float = 3e-4
    enc_lr_scale: float = 0.3
    entropy_coef: float = 1e-4
    grad_clip_norm: float = 20.0
    tau: float = 0.01
    rho: float = 0.5
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    termination_coef: float = 1.0
    discount_denom: float = 5.0
    discount_min: float = 0.95
    discount_max: float = 0.995
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    # Official model_size=5 architecture; exact parameter count depends on obs/action size.
    model_size: int = 5
    num_enc_layers: int = 2
    enc_dim: int = 256
    mlp_dim: int = 512
    latent_dim: int = 512
    num_q: int = 5
    dropout: float = 0.01
    simnorm_dim: int = 8
    num_channels: int = 32
    task_dim: int = 0
    multitask: bool = False
    obs: str = "state"
    episodic: bool = True
    mpc: bool = True
    # Eager first: compilation is an execution optimization, not an algorithm substitution.
    compile: bool = False
    device: str = "cpu"
    obs_shape: dict = field(init=False)
    bin_size: float = field(init=False)

    def __post_init__(self):
        self.obs_shape = {"state": (self.obs_dim,)}
        self.bin_size = (self.vmax - self.vmin) / (self.num_bins - 1) if self.num_bins > 1 else 0.0
        for name in (
            "obs_dim",
            "action_dim",
            "num_envs",
            "plan_chunk_size",
            "episode_length",
            "batch_size",
            "horizon",
            "num_samples",
            "num_elites",
            "iterations",
            "latent_dim",
            "num_q",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.num_pi_trajs < self.num_samples or self.num_elites > self.num_samples:
            raise ValueError("Require 0 <= num_pi_trajs < num_samples and num_elites <= num_samples")
        if self.num_q < 2 or self.latent_dim % self.simnorm_dim:
            raise ValueError("Require at least two Q networks and latent_dim divisible by simnorm_dim")
        if self.multitask or self.task_dim or self.obs != "state" or not self.mpc:
            raise ValueError("This adapter supports single-task state-based TD-MPC2 with MPC enabled")
        if self.model_size != 5:
            raise ValueError("Only the official 5M architecture preset is exposed by this adapter")
        if not 0 < self.min_std <= self.max_std or self.temperature <= 0:
            raise ValueError("Invalid planner standard deviation or temperature")


# Short alias for callers that do not need the algorithm name repeated.
Config = TDMPC2Config
