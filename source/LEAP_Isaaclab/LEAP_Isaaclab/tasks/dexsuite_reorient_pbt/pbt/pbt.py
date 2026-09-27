"""Application compatibility boundary; the PBT engine lives in async-pbt."""

from async_pbt.integrations.rl_games.pbt import (
    MultiObserver,
)
from async_pbt.integrations.rl_games.pbt import PbtAlgoObserver as PublicPbtAlgoObserver
from async_pbt.integrations.rl_games.pbt import initialize_population as public_initialize_population

__all__ = ["MultiObserver", "PbtAlgoObserver", "initialize_population"]


def _normalize(params):
    """Materialize the legacy objective before either public entrypoint reads it."""
    params["pbt"].setdefault("objective", "dexpbt")
    return params


def initialize_population(params, resumed=False):
    return public_initialize_population(_normalize(params), resumed=resumed)


class PbtAlgoObserver(PublicPbtAlgoObserver):
    """Preserve LEAP's workspace and raw-checkpoint launch-parameter contracts."""

    def __init__(self, params, args_cli):
        super().__init__(_normalize(params), args_cli)

    def after_init(self, algo):
        # Public directory is a base; historically LEAP's base is algo.train_dir.
        # Only change the observer-owned cfg temporarily, never the logged config.
        directory = self.cfg.directory
        self.cfg.directory = ""
        resume = self.resume
        # An empty companion disables public preserve_raw without restoring age.
        # plain_checkpoint was already computed by __init__, so raw RNG reseeding
        # and the public restore/evidence wrapper order remain unchanged.
        if resume is None:
            self.resume = {}
        try:
            super().after_init(algo)
        finally:
            self.cfg.directory = directory
            self.resume = resume
