"""CPU-only tests of the trainer-independent TD-MPC2 population boundary."""

import multiprocessing
import sys
import torch
import yaml
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tdmpc2.pbt import PopulationMember, default_pbt_cfg, load_checkpoint

HPARAMS = {"lr": 3e-4, "entropy_coef": 1e-4, "grad_clip_norm": 20.0}
CONTRACT = {"task": "Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0", "num_envs": 256, "horizon": 3}


def member(workspace, policy=0, **overrides):
    cfg = default_pbt_cfg(policy_idx=policy, num_policies=2, **overrides)
    return PopulationMember(workspace, cfg, 42 + policy, HPARAMS, CONTRACT)


def learner(value=1.0):
    return {"agent": {"weight": torch.tensor([value])}, "replay": {"data": torch.tensor([value])}, "trainer": {}}


def test_publication_is_full_atomic_checkpoint_with_monotonic_boundary(tmp_path):
    population = member(tmp_path)
    assert not population.should_publish(999_936)
    assert population.should_publish(1_000_192)
    record = population.publish(learner(), 1_000_192, 37, {"curriculum": 0.2})
    restored = load_checkpoint(record["checkpoint"])
    assert record["algorithm"] == "tdmpc2"
    assert record["schema_version"] == 2
    assert record["params"] == HPARAMS
    assert record["epoch"] == record["updates"] == 37
    assert restored["training_state"]["agent"]["weight"].item() == 1.0
    assert restored["training_state"]["replay"]["data"].item() == 1.0
    assert restored["env_state"] == {"curriculum": 0.2}
    assert restored["pbt"]["frame"] == 1_000_192
    assert Path(record["pbt_checkpoint"]).is_file()
    assert not population.should_publish(1_999_872)
    assert population.should_publish(2_000_128)


def completed(successes, *, gravity=-9.81):
    return {
        "dexpbt": {
            "schema_version": 1,
            "protocol": "training_conditions_fixed_definition_v1",
            "groups": [
                {
                    "condition": {
                        "curriculum_tolerance": 0.1,
                        "position_tolerance": 0.1,
                        "rotation_tolerance": 0.5,
                        "contact_threshold": 0.01,
                        "adr_parameters": {"gravity_z": gravity},
                    },
                    "completed_episodes": 128,
                    "success_count": successes,
                    "position_error_sum": 1.0,
                    "rotation_error_sum": 1.0,
                    "good_contact_fraction_sum": 128.0,
                }
            ],
        }
    }


def optimized_learner(value):
    parameter = torch.nn.Parameter(torch.tensor([value]))
    optimizer = torch.optim.Adam([parameter], lr=3e-4)
    parameter.square().sum().backward()
    optimizer.step()
    return {
        "agent": {"weight": parameter.detach(), "optimizer": optimizer.state_dict()},
        "replay": {"data": torch.arange(12).reshape(4, 3), "cursor": 3},
        "trainer": {"frame": 5_900_032, "updates": 100},
    }


def publish_donor(workspace):
    donor = member(Path(workspace), 0)
    donor.observe(completed(128), 5_900_032)
    donor.publish(optimized_learner(8.0), 5_900_032, 100, {"curriculum": 0.1})


def restore_recipient(workspace):
    recipient = member(Path(workspace), 1)
    recipient.observe(completed(0), 6_000_128)
    recipient_state = optimized_learner(-2.0)
    recipient_state["trainer"] = {"frame": 6_000_128, "updates": 47, "pending_updates": 2, "schedule": {"first": False}}
    own = recipient.publish(recipient_state, 6_000_128, 47, {"curriculum": 0.1})
    decision = recipient.maybe_exploit(own)
    assert decision is not None
    assert decision.frame == 6_000_128 and decision.updates == 47
    assert decision.training_state["trainer"] == {"frame": 5_900_032, "updates": 100}
    expected = optimized_learner(8.0)
    torch.testing.assert_close(decision.training_state["agent"]["weight"], expected["agent"]["weight"])
    assert decision.training_state["agent"]["optimizer"]["state"][0]["step"].item() == 1
    torch.testing.assert_close(
        decision.training_state["agent"]["optimizer"]["state"][0]["exp_avg"],
        expected["agent"]["optimizer"]["state"][0]["exp_avg"],
    )
    torch.testing.assert_close(decision.training_state["replay"]["data"], expected["replay"]["data"])
    assert decision.training_state["replay"]["cursor"] == 3
    assert decision.metadata["recipient_trainer_state"] == recipient_state["trainer"]
    assert decision.metadata["trainer_state"] == {"frame": 5_900_032, "updates": 100}
    assert decision.hparams != HPARAMS
    for name, value in decision.hparams.items():
        low, high = recipient.cfg.mutation_bounds[name]
        assert low <= value <= high
    # Main's explicit restore callback receives full donated state, while its
    # budget comes from decision.frame/updates, not the donated trainer block.
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter])
    parameter.data.copy_(decision.training_state["agent"]["weight"])
    optimizer.load_state_dict(decision.training_state["agent"]["optimizer"])
    assert parameter.item() > 7.9
    assert optimizer.state[parameter]["step"].item() == 1
    next_record = recipient.publish(decision.training_state, 6_000_384, 48, {}, final=True)
    assert next_record["evaluations"] == {}
    assert recipient.maybe_exploit(next_record) is None
    assert load_checkpoint(decision.metadata["checkpoint"])["pbt"]["recipient_frame"] == 6_000_128


def test_two_real_process_members_transfer_full_state_without_donating_budget(tmp_path):
    context = multiprocessing.get_context("spawn")
    for target in (publish_donor, restore_recipient):
        process = context.Process(target=target, args=(str(tmp_path),))
        process.start()
        process.join(timeout=40)
        if process.is_alive():
            process.terminate()
            process.join()
            raise AssertionError("CPU population process timed out")
        assert process.exitcode == 0


def test_pruning_keeps_only_two_owned_archives_and_preserves_foreign_files(tmp_path):
    population = member(tmp_path)
    foreign = population.policy_dir / "00000000000000000001.yaml"
    foreign.write_text("algorithm: other\n")
    foreign.with_suffix(".pth").write_bytes(b"not ours")
    paths = []
    for frame in range(1, 6):
        record = population.publish(learner(), frame, frame, {}, final=frame == 5)
        paths.append(Path(record["checkpoint"]))
    assert not any(path.exists() for path in paths[:3])
    assert all(path.exists() and path.with_suffix(".yaml").exists() for path in paths[3:])
    assert foreign.exists() and foreign.with_suffix(".pth").read_bytes() == b"not ours"
    assert population.maybe_exploit(record) is None


@pytest.mark.parametrize("mismatch", ["algorithm", "contract", "physical", "stale", "insufficient", "warmup", "final"])
def test_selection_fails_closed_before_copying_incompatible_or_unready_donor(tmp_path, mismatch):
    donor = member(tmp_path, 0)
    recipient = member(tmp_path, 1)
    frame = 1_000_192 if mismatch == "warmup" else 6_000_128
    donor.observe(completed(128, gravity=-1.0 if mismatch == "physical" else -9.81), frame)
    donor_record = donor.publish(learner(8.0), frame, 100, {})
    if mismatch in {"algorithm", "contract", "stale"}:
        if mismatch == "algorithm":
            donor_record["algorithm"] = "ppo"
        elif mismatch == "contract":
            donor_record["contract"] = {**CONTRACT, "horizon": 99}
        else:
            donor_record["created_at"] -= 1801
        Path(donor_record["pbt_checkpoint"]).write_text(yaml.safe_dump(donor_record))
    evidence = completed(0)
    if mismatch == "insufficient":
        evidence["dexpbt"]["groups"][0].update(completed_episodes=127, good_contact_fraction_sum=127)
    recipient.observe(evidence, frame)
    own = recipient.publish(learner(), frame, 47, {}, final=mismatch == "final")
    assert recipient.maybe_exploit(own) is None
    assert not (recipient.policy_dir / "restart" / "checkpoint.pth").exists()


def test_resume_restores_recipient_rng_counters_and_requires_fresh_evidence(tmp_path):
    donor = member(tmp_path, 0)
    recipient = member(tmp_path, 1)
    frame = 6_000_128
    donor.observe(completed(128), frame)
    donor_record = donor.publish(learner(8.0), frame, 100, {})
    recipient.observe(completed(0), frame)
    own = recipient.publish(learner(), frame, 47, {})
    decision = recipient.maybe_exploit(own)
    assert decision is not None
    restored = load_checkpoint(decision.metadata["checkpoint"])
    assert restored["pbt"]["frame"] == frame
    assert restored["pbt"]["epoch"] == restored["pbt"]["updates"] == 47
    assert restored["pbt"]["pbt_rng_state"] != donor_record["pbt_rng_state"]
    resumed = member(tmp_path, 1)
    resumed.observe(completed(128), frame)
    resumed.restore_metadata(restored["pbt"])
    assert resumed.hparams == decision.hparams
    assert resumed.frame == frame and resumed.updates == 47
    assert not resumed.should_publish(frame + 256)
    cleared = resumed.publish(learner(8.0), frame + 256, 48, {})
    assert cleared["evaluations"] == {}
    assert cleared["pbt_rng_state"] == decision.metadata["pbt_rng_state"]
    resumed.observe(completed(0), frame + 512)
    assert resumed.maybe_exploit(resumed.publish(learner(), frame + 512, 49, {})) is None
    regular = member(tmp_path, 1)
    regular.restore_metadata(load_checkpoint(cleared["checkpoint"])["pbt"])
    assert regular.frame == frame + 256


def test_corrupted_donor_snapshot_skips_replacement_without_advancing_rng(tmp_path):
    donor = member(tmp_path, 0)
    recipient = member(tmp_path, 1)
    frame = 6_000_128
    donor.observe(completed(128), frame)
    published = donor.publish(learner(8.0), frame, 100, {})
    Path(published["checkpoint"]).write_bytes(b"invalid checkpoint data")
    recipient.observe(completed(0), frame)
    own = recipient.publish(learner(), frame, 47, {})
    assert recipient.maybe_exploit(own) is None
    after = recipient.publish(learner(), frame + 256, 48, {})
    assert after["pbt_rng_state"] == own["pbt_rng_state"]
    assert after["params"] == HPARAMS
    assert after["evaluations"]


def test_record_identity_prevents_loading_a_different_snapshot(tmp_path):
    population = member(tmp_path)
    first = population.publish(learner(1.0), 256, 1, {})
    second = population.publish(learner(2.0), 512, 2, {})
    Path(first["checkpoint"]).write_bytes(Path(second["checkpoint"]).read_bytes())
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(first["checkpoint"])


def test_donor_pruning_between_selection_and_copy_skips_without_consuming_evidence(tmp_path):
    donor = member(tmp_path, 0)
    recipient = member(tmp_path, 1)
    frame = 6_000_128
    donor.observe(completed(128), frame)
    published = donor.publish(learner(8.0), frame, 100, {})
    events = []

    def on_log(event):
        events.append(event)
        if event.get("reason") == "selected":
            Path(published["checkpoint"]).unlink()

    recipient.log_callback = on_log
    recipient.observe(completed(0), frame)
    own = recipient.publish(learner(), frame, 47, {})
    assert recipient.maybe_exploit(own) is None
    assert events[-1]["event"] == "replacement_skipped"
    later = recipient.publish(learner(), frame + 256, 48, {})
    assert later["pbt_rng_state"] == own["pbt_rng_state"]
    assert later["evaluations"]


def test_rejects_ppo_mutation_configuration_and_invalid_hyperparameters(tmp_path):
    cfg = default_pbt_cfg(mutation={"kl_threshold": "mutate_float"}, mutation_bounds={"kl_threshold": (0.001, 0.1)})
    with pytest.raises(ValueError, match="TD-MPC2"):
        PopulationMember(tmp_path, cfg, 42, HPARAMS, CONTRACT)
    with pytest.raises(ValueError, match="hyperparameter"):
        PopulationMember(tmp_path, default_pbt_cfg(), 42, {**HPARAMS, "lr": float("nan")}, CONTRACT)
