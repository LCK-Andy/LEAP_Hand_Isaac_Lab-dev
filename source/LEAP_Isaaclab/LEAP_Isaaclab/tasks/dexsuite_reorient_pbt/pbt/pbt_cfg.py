"""Legacy application defaults over the public validated configuration."""

from dataclasses import dataclass

from async_pbt.pbt_cfg import PbtCfg as PublicPbtCfg


@dataclass
class PbtCfg(PublicPbtCfg):
    """Keep Dexsuite scoring as the historical implicit objective."""

    objective: str = "dexpbt"
