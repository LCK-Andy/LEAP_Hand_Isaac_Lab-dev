"""Compatibility exports delegated to the installed async-pbt package."""

from async_pbt.adapters.allegro import (
    GoalWindow,
    attach_measurement,
    select_goal_donor,
)

__all__ = ["GoalWindow", "select_goal_donor", "attach_measurement"]
