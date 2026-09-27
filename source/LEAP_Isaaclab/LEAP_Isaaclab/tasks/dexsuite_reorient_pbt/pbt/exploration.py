"""Compatibility exports delegated to the installed async-pbt package."""

from async_pbt.integrations.rl_games.exploration import (
    ExplorationObserver,
)

__all__ = ["ExplorationObserver"]
