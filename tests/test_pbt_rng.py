from async_pbt.integrations.rl_games import pbt as public_pbt

"""Recipient-owned PBT randomness across public publication/restart boundaries."""

import copy
import json
import random
import types
import yaml

import pytest
from test_dexpbt_pbt import FakeAlgo, config, episode_info, load_module


def observer_at(pbt, root, member, metadata=None):
    cfg = config(member)
    if metadata is not None:
        load_module("runtime").apply_overrides(cfg, metadata["params"])
    args = types.SimpleNamespace(
        original_argv=["train.py"],
        pbt_resume_metadata=metadata,
        checkpoint=(metadata["checkpoint"] if metadata and "checkpoint" in metadata else None),
    )
    observer = pbt.PbtAlgoObserver(cfg, args)
    algo = FakeAlgo(root, cfg, f"member-{member}")
    observer.after_init(algo)
    if metadata is not None:
        algo.restore(metadata["checkpoint"])
    return observer, algo


def publish(observer, algo, frame, success=None):
    algo.frame, algo.epoch_num = frame, frame // 100
    if success is not None:
        observer.process_infos(episode_info(success=success), [0, 1, 2, 3])
    observer.after_print_stats(frame, algo.epoch_num, 1.0)


def restart_record(observer):
    with open(observer.curr_policy_dir + "/restart/checkpoint.yaml") as stream:
        return yaml.safe_load(stream)


def test_global_reseeding_does_not_change_recipient_mutation(tmp_path, monkeypatch):
    pbt = load_module("pbt")
    monkeypatch.setattr(public_pbt.os, "execv", lambda *_: None)
    outcomes = []
    for branch, global_seed in enumerate((123, 999)):
        root = tmp_path / str(branch)
        donor, donor_algo = observer_at(pbt, root, 0)
        publish(donor, donor_algo, 100, success=4)
        recipient, algo = observer_at(pbt, root, 1)
        random.seed(global_seed)  # RL/Isaac startup and checkpoint restore can reseed this.
        for _ in range(branch * 17):
            random.random()
        before = random.getstate()
        publish(recipient, algo, 200, success=0)
        outcomes.append(restart_record(recipient)["params"])
        assert random.getstate() == before
    assert outcomes[0] == outcomes[1]


def test_replacement_and_publication_resume_continue_recipient_not_donor_rng(tmp_path, monkeypatch):
    pbt = load_module("pbt")
    runtime = load_module("runtime")
    monkeypatch.setattr(public_pbt.os, "execv", lambda *_: None)
    donor, donor_algo = observer_at(pbt, tmp_path, 0)
    publish(donor, donor_algo, 100, success=4)
    donor_record = runtime.resume_metadata(donor.curr_policy_dir + "/000000000001.pth")
    recipient, algo = observer_at(pbt, tmp_path, 1)
    publish(recipient, algo, 200, success=0)
    first = restart_record(recipient)
    before_draws = runtime.resume_metadata(recipient.curr_policy_dir + "/000000000002.pth")
    assert first["pbt_rng_state"] != before_draws["pbt_rng_state"]
    assert first["pbt_rng_state"] != donor_record["pbt_rng_state"]
    # Both the staged and ordinary companion must survive safe YAML and JSON codecs.
    first = json.loads(json.dumps(first))
    publish(recipient, algo, 300)
    ordinary = runtime.resume_metadata(recipient.curr_policy_dir + "/000000000003.pth")
    assert ordinary["pbt_rng_state"] == first["pbt_rng_state"]
    publish(recipient, algo, 400, success=0)
    uninterrupted = restart_record(recipient)
    assert uninterrupted["params"] != first["params"]
    for metadata in (first, ordinary):
        random.seed(42)  # Re-exec reuses the recipient's fixed seed.
        restored, restored_algo = observer_at(pbt, tmp_path, 1, metadata)
        random.seed(9876)  # A donor PPO restore must not own PBT randomness.
        publish(restored, restored_algo, 400, success=0)
        resumed = restart_record(restored)
        assert resumed["params"] == uninterrupted["params"]
        assert resumed["pbt_rng_state"] == uninterrupted["pbt_rng_state"]


@pytest.mark.parametrize("state", [None, [], [3, [], None], [99, [], None], "not a state"])
def test_invalid_present_rng_state_fails_clearly(tmp_path, state):
    pbt = load_module("pbt")
    with pytest.raises(ValueError, match="Invalid PBT RNG state"):
        observer_at(pbt, tmp_path, 1, {"params": {}, "pbt_rng_state": state})


def test_legacy_fallback_is_warned_reproducible_and_recipient_age_owned(tmp_path):
    pbt = load_module("pbt")
    utils = load_module("pbt_utils")
    streams = []
    for global_seed, donor_frame in ((1, 100), (2, 9999)):
        random.seed(global_seed)
        args = types.SimpleNamespace(
            original_argv=["train.py"],
            pbt_resume_metadata={
                "frame": donor_frame,
                "recipient_frame": 400,
                "recipient_epoch": 4,
            },
        )
        with pytest.warns(RuntimeWarning, match="legacy.*pbt_rng_state.*pbt_rng_legacy_v1"):
            observer = pbt.PbtAlgoObserver(config(1), args)
        streams.append(utils.rng_state(observer.rng))
    assert streams[0] == streams[1]
    args.pbt_resume_metadata["recipient_frame"] = 500
    with pytest.warns(RuntimeWarning, match="legacy"):
        later = pbt.PbtAlgoObserver(config(1), args)
    assert utils.rng_state(later.rng) != streams[0]
    fresh, _ = observer_at(pbt, tmp_path, 1)
    assert utils.rng_state(fresh.rng) != streams[0]
    with pytest.warns(RuntimeWarning, match="legacy"):
        other = pbt.PbtAlgoObserver(config(0), args)
    assert utils.rng_state(other.rng) != utils.rng_state(later.rng)


def test_staging_legacy_recipient_does_not_import_donor_rng(tmp_path):
    utils = load_module("pbt_utils")
    runtime = load_module("runtime")
    checkpoint = tmp_path / "donor.pth"
    checkpoint.write_bytes(b"donor")
    donor = {
        "checkpoint": str(checkpoint),
        "policy_idx": 0,
        "frame": 100,
        "pbt_rng_state": ["donor"],
    }
    recipient = {"policy_idx": 1, "frame": 200, "epoch": 2}
    staged, companion = utils.stage_restart(tmp_path, donor, recipient, {})
    assert "pbt_rng_state" not in runtime.resume_metadata(staged, companion)


def test_plain_checkpoint_fallback_uses_restored_age_and_is_published(tmp_path):
    import torch

    pbt = load_module("pbt")
    utils = load_module("pbt_utils")
    runtime = load_module("runtime")
    checkpoint = tmp_path / "plain.pth"
    source = FakeAlgo(tmp_path, config(1), "SOURCE")
    states = []
    for frame, global_seed in ((400, 12), (400, 99), (500, 12)):
        source.frame, source.epoch_num = frame, frame // 100
        torch.save(source.get_full_state_weights(), checkpoint)
        args = types.SimpleNamespace(original_argv=["train.py"], checkpoint=str(checkpoint))
        observer = pbt.PbtAlgoObserver(config(1), args)
        algo = FakeAlgo(tmp_path, config(1), "RESTORED")
        observer.after_init(algo)
        random.seed(global_seed)
        global_before = random.getstate()
        with pytest.warns(RuntimeWarning, match=f"pbt_rng_legacy_v1:42:1:{frame}:"):
            algo.restore(str(checkpoint))
        assert random.getstate() == global_before
        states.append(utils.rng_state(observer.rng))
        publish(observer, algo, frame + 100)
        metadata = runtime.resume_metadata(observer.curr_policy_dir + f"/{frame // 100 + 1:012d}.pth")
        assert metadata["pbt_rng_state"] == states[-1]
    assert states[0] == states[1]
    assert states[1] != states[2]
    fresh, _ = observer_at(pbt, tmp_path, 1)
    assert utils.rng_state(fresh.rng) != states[0]


def test_copy_only_bypasses_mutation_and_keeps_recipient_rng(tmp_path, monkeypatch):
    pbt = load_module("pbt")
    runtime = load_module("runtime")
    monkeypatch.setattr(public_pbt.os, "execv", lambda *_: None)
    donor, donor_algo = observer_at(pbt, tmp_path, 0)
    donor_algo.last_lr = 0.02  # copy-only must not even clamp adaptive LR
    publish(donor, donor_algo, 100, success=4)
    recipient, algo = observer_at(pbt, tmp_path, 1)
    recipient.cfg.copy_only = True
    publish(recipient, algo, 200, success=0)
    before = runtime.resume_metadata(recipient.curr_policy_dir + "/000000000002.pth")
    staged = restart_record(recipient)
    assert staged["params"]["agent.params.config.learning_rate"] == 0.02
    assert staged["pbt_rng_state"] == before["pbt_rng_state"]


def test_mutation_arms_preserve_realized_initial_configuration():
    pbt = load_module("pbt")
    values = []
    for knobs in (
        {},
        {"mutation_rate": 1.0},
        {"copy_only": True},
        {"change_range": [1.05, 1.2]},
    ):
        settings = config(1)
        settings["pbt"].update(objective="allegro_goals", **knobs)
        values.append(pbt.initialize_population(settings))
    assert all(value == values[0] for value in values)


def test_initial_population_diversity_keeps_seed_pairs_and_baseline():
    pbt = load_module("pbt")
    original = config(0)
    baseline = copy.deepcopy(original)
    before = random.getstate()
    pbt.initialize_population(baseline)
    assert baseline == {**original, "pbt": {**original["pbt"], "objective": "dexpbt"}}
    first, repeated = config(1), config(1)
    assert pbt.initialize_population(first) == pbt.initialize_population(repeated)
    assert first["params"]["config"] != original["params"]["config"]
    assert random.getstate() == before
