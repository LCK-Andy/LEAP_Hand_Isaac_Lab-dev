"""Compatibility exports delegated to the installed async-pbt package."""

from async_pbt.integrations.rl_games.checkpoint import save_pbt_checkpoint
from async_pbt.pbt_utils import EpisodeWindow as PublicEpisodeWindow
from async_pbt.pbt_utils import (
    atomic_yaml,
    cleanup,
    eligible,
    filter_params,
    flatten_dict,
    load_pbt_ckpts,
    recipient_rng,
    rng_state,
    select_donor,
    stage_restart,
)

from .pbt_cfg import PbtCfg


class EpisodeWindow(PublicEpisodeWindow):
    """Retain the legacy no-argument Dexsuite accumulator default."""

    def __init__(self, cfg=None):
        super().__init__(cfg if cfg is not None else PbtCfg())


__all__ = [
    "rng_state",
    "recipient_rng",
    "flatten_dict",
    "filter_params",
    "atomic_yaml",
    "load_pbt_ckpts",
    "cleanup",
    "stage_restart",
    "EpisodeWindow",
    "eligible",
    "select_donor",
    "save_pbt_checkpoint",
]
