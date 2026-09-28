"""Simulator-free oracles for the isolated original population recipe."""

import ast
import copy
import random
import sys
import yaml
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Standalone test invocation requires the repository path bootstrap before local imports.
from scripts.dexpbt_original.pbt import (  # noqa: E402
    ObjectiveWindow,
    OriginalPbtObserver,
    PbtConfig,
    atomic_json,
    cleanup_history,
    load_population,
    mutate,
    population_contract,
    restart_state,
    select,
)
from scripts.dexpbt_original.runtime import (  # noqa: E402
    ENV_DEFAULTS,
    MUTATIONS,
    PPO_PREFIX,
    SCHEMA,
    TASK,
    UPSTREAM,
    mutable_values,
)

FIXTURE = ROOT / "tests/fixtures/dexpbt_original/isaacgymenvs"
AGENT = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexpbt_original/agents/rl_games_ppo.yaml"


def values():
    return mutable_values(yaml.safe_load(AGENT.read_text()), ENV_DEFAULTS)


def record(index, objective, iteration=10):
    return {"member": index, "objective": objective, "mutable": values(), "iteration": iteration}


def state(member=0, frame=100, epoch=10):
    return {
        "model": {"weight": member},
        "optimizer": {"parent": member},
        "env_state": {"success_tolerance": 0.02},
        "frame": frame,
        "epoch": epoch,
        "dexpbt_original": {
            "schema": SCHEMA,
            "task": TASK,
            "upstream": UPSTREAM,
            "member": member,
            "stamp": "same",
            "profile": "released_source",
            "mutable": values(),
            "rng_state": random.Random(member).getstate(),
            "lineage": [],
            "segment_start_frame": 0,
        },
    }


def test_source_mutation_oracle():
    # Execute only pure mutation functions from the immutable source fixture.
    source = FIXTURE / "pbt/mutation.py"
    tree = ast.parse(source.read_text())
    safe = ast.Module(
        body=[node for node in tree.body if isinstance(node, (ast.Import, ast.FunctionDef))], type_ignores=[]
    )
    namespace = {}
    exec(compile(safe, str(source), "exec"), namespace)
    functions = {
        "float": "mutate_float",
        "clip": "mutate_eps_clip",
        "integer": "mutate_mini_epochs",
        "discount": "mutate_discount",
    }
    mapping = {key: functions[kind] for key, kind in MUTATIONS.items()}
    for seed in range(30):
        namespace["random"] = random.Random(seed)
        expected = namespace["mutate"](values(), mapping, 0.15, 1.1, 1.5)
        assert mutate(values(), 0.15, random.Random(seed)) == expected


def test_recurrent_config_matches_merged_source():
    original = yaml.safe_load((FIXTURE / "cfg/train/AllegroKukaPPO.yaml").read_text())["params"]
    lstm = yaml.safe_load((FIXTURE / "cfg/train/AllegroKukaLSTMPPO.yaml").read_text())["params"]
    local = yaml.safe_load(AGENT.read_text())["params"]
    for name in ("algo", "model"):
        assert local[name] == original[name]
    network = copy.deepcopy(original["network"])
    network.update(lstm["network"])
    assert local["network"] == network
    config = {**original["config"], **lstm["config"]}
    for name in (
        "gamma",
        "tau",
        "learning_rate",
        "lr_schedule",
        "schedule_type",
        "kl_threshold",
        "entropy_coef",
        "grad_norm",
        "e_clip",
        "minibatch_size",
        "mini_epochs",
        "critic_coef",
        "horizon_length",
        "seq_length",
        "bounds_loss_coef",
        "reward_shaper",
        "normalize_input",
        "normalize_value",
        "normalize_advantage",
        "clip_value",
    ):
        expected = float(config[name]) if isinstance(local["config"][name], float) else config[name]
        assert local["config"][name] == expected
    assert local["config"]["torch_compile"] is False
    assert "exploration" not in local["config"]


def test_only_source_present_mutation_keys_are_active():
    declared = yaml.safe_load((FIXTURE / "cfg/pbt/mutation/allegro_kuka_mutation.yaml").read_text())
    task = yaml.safe_load((FIXTURE / "cfg/task/AllegroKuka.yaml").read_text())
    present = {key for key in declared if key.startswith("task.env.") and key.removeprefix("task.env.") in task["env"]}
    assert present == {"task.env." + key for key in ENV_DEFAULTS}
    assert len(present) == 9


def test_profiles_are_explicit_and_distinct():
    source, paper = PbtConfig(), PbtConfig(profile="paper")
    assert (source.mutation_rate, source.adaptation) == (0.15, 100_000_000)
    assert (paper.mutation_rate, paper.adaptation) == (0.2, 50_000_000)
    assert source.initial_delay == paper.initial_delay == 200_000_000
    assert source.interval == paper.interval == 20_000_000
    with pytest.raises(ValueError, match="four"):
        PbtConfig(population=2)


def test_source_middle_and_top_untouched_paper_middle_mutates():
    records = {i: record(i, i + 1) for i in range(8)}
    for member in (3, 4, 7):
        cfg = PbtConfig(member=member)
        assert select(records, cfg, member + 1, member + 1, random.Random(0))[0] is None
    cfg = PbtConfig(profile="paper", member=3)
    assert select(records, cfg, 4, 4, random.Random(0)) == (3, 3, "middle_self_mutation")
    cfg = PbtConfig(profile="paper", member=0)
    donor, parent, reason = select(records, cfg, 1, 1, random.Random(0))
    assert donor == parent and donor in (5, 6, 7) and reason == "bottom_copy_top"


def test_source_margin_failure_still_mutates_self():
    records = {i: record(i, 1 + i * 0.001) for i in range(8)}
    assert select(records, PbtConfig(), 1, 1, random.Random(0)) == (0, 0, "margin_self_mutation")


def test_source_best_protection_and_parameter_parent_coin():
    records = {i: record(i, i + 1) for i in range(8)}
    assert select(records, PbtConfig(), 1, 6, random.Random(0))[2] == "best_in_interval_protection"
    parents = {select(records, PbtConfig(), 1, 1, random.Random(seed))[1] for seed in range(30)}
    assert 0 in parents and any(parent >= 5 for parent in parents)


def test_window_last_completed_per_env_not_mean_of_batch():
    window = ObjectiveWindow(2)
    window.add([1, 999], [0])
    assert window.current is None
    window.add([100, 2], [1])
    assert window.current == 1.5
    window.add([3, 999], [0])
    assert window.current == 2.5 and window.best == 2.5
    window.consume()
    window.add([999, 4], [1])
    assert window.current == 3.5
    with pytest.raises(ValueError, match="Nonfinite"):
        window.add([float("nan"), 1], [0])


def test_historical_selection_never_reads_future_or_latest_only(tmp_path):
    cfg = PbtConfig()
    directory = tmp_path / "001"
    directory.mkdir()
    for iteration in (1, 5, 10):
        path = directory / f"{iteration:09d}.pth"
        path.touch()
        atomic_json(
            path.with_suffix(".json"),
            {
                **record(1, iteration, iteration),
                "checkpoint": str(path),
                "stamp": "same",
                "profile": cfg.profile,
                "population_contract": population_contract(cfg),
            },
        )
    chosen = load_population(tmp_path, cfg, 6, "same")
    assert chosen[1]["iteration"] == 5
    assert load_population(tmp_path, cfg, 0, "same") == {}
    with pytest.raises(ValueError, match="cadence"):
        load_population(tmp_path, PbtConfig(interval=40_000_000), 6, "same")
    with pytest.raises(ValueError, match="Incompatible"):
        load_population(tmp_path, cfg, 6, "different")


def test_cleanup_keeps_sparse_old_history_and_latest_ten(tmp_path):
    for iteration in range(1, 41):
        (tmp_path / f"{iteration:09d}.json").write_text("{}")
        (tmp_path / f"{iteration:09d}.pth").touch()
    cleanup_history(tmp_path, {0: {"iteration": 0}})
    kept = sorted(int(p.stem) for p in tmp_path.glob("*.json"))
    assert len(kept) == 25 and kept[0] == 1 and kept[-10:] == list(range(31, 41))


def test_restart_keeps_donor_training_state_recipient_age_and_mutations():
    donor, own = state(7, 50, 5), state(0, 100, 10)
    mutated = values()
    mutated[PPO_PREFIX + "mini_epochs"] = 3
    mutated["task.env.liftingBonus"] = 450.0
    rng = random.Random(99).getstate()
    output = restart_state(donor, own, mutated, rng)
    assert output["model"] == donor["model"] and output["optimizer"] == donor["optimizer"]
    assert output["env_state"] == donor["env_state"]
    assert (output["frame"], output["epoch"]) == (100, 10)
    assert output["dexpbt_original"]["member"] == 0
    assert output["dexpbt_original"]["mutable"] == mutated
    assert output["dexpbt_original"]["rng_state"] == rng
    assert output["dexpbt_original"]["segment_start_frame"] == 100
    assert donor["frame"] == 50


@pytest.mark.parametrize(
    "env_state",
    [
        {"tolerance": 0.02},
        {},
        {"success_tolerance": float("nan")},
        {"success_tolerance": float("inf")},
        {"success_tolerance": 0.0},
        {"success_tolerance": 0.08},
        {"success_tolerance": True},
    ],
)
def test_invalid_checkpoint_tolerance_is_refused(env_state):
    from scripts.dexpbt_original.runtime import validate_checkpoint

    saved = state()
    saved["env_state"] = env_state
    with pytest.raises(ValueError, match="success_tolerance"):
        validate_checkpoint(saved)


def test_disabled_pbt_still_publishes_recoverable_periodic_checkpoint(tmp_path):
    from types import SimpleNamespace

    saved = state(frame=100, epoch=10)
    cfg = PbtConfig(enabled=False, interval=50, minimum_runtime_seconds=0)
    observer = OriginalPbtObserver(cfg, tmp_path, values(), "same", 2, 42, ["train.py"])
    observer.after_init(
        SimpleNamespace(
            frame=100,
            epoch_num=10,
            max_frames=1000,
            max_epochs=100,
            get_full_state_weights=lambda: copy.deepcopy(saved),
            writer=SimpleNamespace(add_scalar=lambda *args: None),
        )
    )
    observer.after_print_stats(50, 10, 0)
    assert (tmp_path / "000" / "000000002.pth").is_file()


def test_missing_true_objective_is_error_even_without_pbt(tmp_path):
    observer = OriginalPbtObserver(PbtConfig(enabled=False), tmp_path, values(), "same", 2, 42, ["train.py"])
    with pytest.raises(ValueError, match="reward fallback"):
        observer.process_infos({}, None)
