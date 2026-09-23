"""Validated simulator-independent PBT configuration."""

from dataclasses import dataclass, field


@dataclass
class PbtCfg:
    enabled: bool = False
    policy_idx: int = 0
    num_policies: int = 4
    directory: str = ""
    workspace: str = "pbt_workspace"
    objective: str = "dexpbt"
    interval_steps: int = 1_000_000
    threshold_std: float = 0.2
    threshold_abs: float = 0.01
    mutation_rate: float = 0.25
    change_range: tuple = (1.1, 2.0)
    mutation: dict = field(default_factory=dict)
    mutation_bounds: dict = field(default_factory=dict)
    initial_diversity: bool = True
    min_completed_episodes: int = 128
    max_score_age_seconds: float = 1800.0
    max_frame_lag: int = 2_000_000
    warmup_steps: int = 0
    adaptation_steps: int = 0
    progress_weight: float = 0.05
    position_error_scale: float = 1.0
    fixed_evaluation_adr_fraction: float | None = None

    def __post_init__(self):
        if not 0 <= self.policy_idx < self.num_policies:
            raise ValueError("policy_idx must be inside the configured population")
        if self.objective != "dexpbt":
            raise ValueError("PBT requires completed-episode dexpbt statistics, never soft reward")
        if self.interval_steps <= 0 or self.min_completed_episodes <= 0:
            raise ValueError("PBT interval and episode window must be positive")
        if self.max_score_age_seconds <= 0 or self.max_frame_lag < 0:
            raise ValueError("PBT freshness limits must be positive/nonnegative")
        if not 0 <= self.mutation_rate <= 1 or not 1 < self.change_range[0] <= self.change_range[1]:
            raise ValueError("Invalid mutation probability/factor range")
        if self.position_error_scale <= 0 or not 0 < self.progress_weight <= 1:
            raise ValueError("Invalid normalized progress scaling")
        if min(self.warmup_steps, self.adaptation_steps, self.threshold_std, self.threshold_abs) < 0:
            raise ValueError("PBT delays and selection margins must be nonnegative")
        for key in self.mutation:
            if key not in self.mutation_bounds:
                raise ValueError(f"Explicit mutation bounds required: {key}")
