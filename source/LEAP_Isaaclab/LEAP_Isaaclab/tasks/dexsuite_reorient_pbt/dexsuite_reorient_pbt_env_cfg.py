"""Official DexSuite Reorient MDP with initialized, checkpointable tolerance curriculum."""

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg import (
    DexsuiteKukaAllegroReorientEnvCfg,
    DexsuiteKukaAllegroReorientEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.curriculums import DifficultyScheduler

from .evaluation import anneal_success_tolerance


class CheckpointableDifficultyScheduler(DifficultyScheduler):
    """Keep official ADR updates; freeze only for explicit evaluation/non-episode resets."""

    def __call__(self, env, env_ids, init_difficulty=0, min_difficulty=0, max_difficulty=50, promotion_only=False):
        fixed = getattr(env, "_dexpbt_fixed_adr", None)
        if fixed is not None:
            self.current_adr_difficulties.fill_(fixed * max_difficulty)
        elif getattr(env, "_dexpbt_in_step", False):
            return super().__call__(env, env_ids, init_difficulty, min_difficulty, max_difficulty, promotion_only)
        self.difficulty_frac = self.current_adr_difficulties.mean() / max(max_difficulty, 1)
        return self.difficulty_frac


@configclass
class DexsuiteReorientPbtEnvCfg(DexsuiteKukaAllegroReorientEnvCfg):
    """Official task plus tolerance annealing; reward weights and MDP are untouched."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.success.params["pos_std"] = 0.2
        self.curriculum.adr.func = CheckpointableDifficultyScheduler
        self.curriculum.anneal_success_tolerance = CurriculumTermCfg(
            func=anneal_success_tolerance,
            params={
                "start_std": 0.2,
                "final_std": 0.1,
                "factor": 0.9,
                "n_succ": 3,
                "rate_threshold": 0.5,
                "window_episodes": 1024,
            },
        )


@configclass
class DexsuiteReorientPbtEnvCfg_PLAY(DexsuiteKukaAllegroReorientEnvCfg_PLAY):
    """Official Play initial difficulty; call configure_fixed_evaluation for heldout runs."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.adr.func = CheckpointableDifficultyScheduler
