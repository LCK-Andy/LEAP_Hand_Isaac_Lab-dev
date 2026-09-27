"""Transition-count scheduling stays independent of vector-environment count."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/tdmpc2/schedule.py"
spec = importlib.util.spec_from_file_location("tdmpc2_schedule", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
UpdateSchedule = module.UpdateSchedule


def test_warmup_crossing_counts_all_seed_transitions_once():
    schedule = UpdateSchedule(seed_steps=1800, updates_per_transition=1.0)
    for _ in range(7):
        assert schedule.collect(256, ready=True) == 0
    assert schedule.collect(256, ready=True) == 2048
    assert schedule.collect(256, ready=True) == 256


def test_fractional_update_credit_survives_checkpoint():
    schedule = UpdateSchedule(seed_steps=0, updates_per_transition=0.1)
    assert schedule.collect(3, ready=True) == 0
    restored = UpdateSchedule(seed_steps=0, updates_per_transition=0.1)
    restored.load_state_dict(schedule.state_dict())
    assert restored.collect(7, ready=True) == 1


def test_replay_readiness_defers_updates_without_losing_credit():
    schedule = UpdateSchedule(seed_steps=4, updates_per_transition=0.5)
    assert schedule.collect(4, ready=False) == 0
    assert schedule.collect(4, ready=True) == 4


def test_schedule_rejects_changed_update_density_on_resume():
    schedule = UpdateSchedule(seed_steps=4, updates_per_transition=1.0)
    with pytest.raises(ValueError, match="configuration"):
        UpdateSchedule(seed_steps=4, updates_per_transition=0.5).load_state_dict(schedule.state_dict())
