"""Validated simulator-independent PBT configuration."""

from dataclasses import dataclass, field


@dataclass
class PbtCfg:
    enabled: bool = False
    selection_enabled: bool = True
    policy_idx: int = 0
    num_policies: int = 4
    directory: str = ""
    workspace: str = "pbt_workspace"
    objective: str = "dexpbt"
    interval_steps: int = 1_000_000
    threshold_std: float | None = None
    threshold_abs: float = 0.01
    contact_progress_threshold_abs: float = 0.00001
    mutation_rate: float | None = None
    change_range: tuple[float, float] | None = None
    copy_only: bool = False
    evidence_windows: int = 1
    recipient_role: str = "any"
    donor_maturity_rollouts: int = 0
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
        if self.mutation_rate is None:
            self.mutation_rate = 0.33 if self.objective == "allegro_goals" else 0.25
        if self.change_range is None:
            self.change_range = (1.1, 1.5) if self.objective == "allegro_goals" else (1.1, 2.0)
        if self.threshold_std is None:
            self.threshold_std = 2.0 if self.objective == "allegro_goals" else 0.2
        if (
            type(self.copy_only) is not bool
            or type(self.evidence_windows) is not int
            or self.evidence_windows not in (1, 2)
        ):
            raise ValueError("copy_only must be boolean and evidence_windows must be 1 or 2")
        if self.recipient_role not in {"any", "weakest"}:
            raise ValueError("recipient_role must be any or weakest")
        if type(self.donor_maturity_rollouts) is not int or self.donor_maturity_rollouts < 0:
            raise ValueError("donor_maturity_rollouts must be a nonnegative integer")
        if self.objective != "allegro_goals" and (
            self.evidence_windows != 1 or self.recipient_role != "any" or self.donor_maturity_rollouts
        ):
            raise ValueError("Evidence/role/maturity pilots currently require allegro_goals")
        if not 0 <= self.policy_idx < self.num_policies:
            raise ValueError("policy_idx must be inside the configured population")
        if self.objective not in {"dexpbt", "allegro_goals", "contact_goals"}:
            raise ValueError("PBT requires completed-episode dexpbt statistics, never soft reward")
        if self.interval_steps <= 0 or self.min_completed_episodes <= 0:
            raise ValueError("PBT interval and episode window must be positive")
        if self.max_score_age_seconds <= 0 or self.max_frame_lag < 0:
            raise ValueError("PBT freshness limits must be positive/nonnegative")
        if not 0 <= self.mutation_rate <= 1 or not 1 < self.change_range[0] <= self.change_range[1]:
            raise ValueError("Invalid mutation probability/factor range")
        if self.position_error_scale <= 0 or not 0 < self.progress_weight <= 1:
            raise ValueError("Invalid normalized progress scaling")
        if (
            min(
                self.warmup_steps,
                self.adaptation_steps,
                self.threshold_std,
                self.threshold_abs,
                self.contact_progress_threshold_abs,
            )
            < 0
        ):
            raise ValueError("PBT delays and selection margins must be nonnegative")
        for key in self.mutation:
            if key not in self.mutation_bounds:
                raise ValueError(f"Explicit mutation bounds required: {key}")
