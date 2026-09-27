"""Checkpoint accounting belongs to the recipient even when learner state transfers."""

import pytest

from scripts.tdmpc2.runtime import canonical_contract, recipient_trainer_state


def test_contract_roundtrip_tuples_match_checkpoint_json():
    assert canonical_contract({"obs_shape": {"state": (1745,)}}) == {"obs_shape": {"state": [1745]}}


def test_staged_restart_uses_recipient_schedule_and_budget():
    recipient = {"frame": 6000, "updates": 50, "schedule": {"frames": 6000, "credit": 0.5}}
    payload = {
        "pbt": {
            "frame": 5000,
            "epoch": 90,
            "recipient_frame": 6000,
            "recipient_epoch": 50,
            "recipient_trainer_state": recipient,
        },
        "training_state": {"trainer": {"frame": 5000, "updates": 90}},
    }
    assert recipient_trainer_state(payload) == recipient
    del payload["pbt"]["recipient_trainer_state"]
    with pytest.raises(ValueError, match="recipient trainer"):
        recipient_trainer_state(payload)


def test_regular_resume_retains_original_accounting():
    state = {"frame": 4096, "updates": 2048}
    assert (
        recipient_trainer_state({"pbt": {"frame": 4096, "epoch": 2048}, "training_state": {"trainer": state}}) == state
    )
