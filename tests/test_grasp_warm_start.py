"""Real RL-Games model and fresh-training warm-start seams; CPU only."""

import torch
import yaml
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.common.a2c_common import A2CBase
from test_dexpbt_pbt import load_module

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_grasp/agents/rl_games_ppo_grasp_cfg.yaml"


def model(dim):
    params = yaml.safe_load(CFG.read_text())["params"]
    return (
        ModelBuilder()
        .load(params)
        .build(
            dict(
                actions_num=16, input_shape=(dim,), num_seqs=1, value_size=1, normalize_input=True, normalize_value=True
            )
        )
    )


def donor():
    net = model(124).eval()
    with torch.no_grad():
        net.running_mean_std.running_mean.copy_(torch.linspace(-1, 1, 124))
        net.running_mean_std.running_var.copy_(torch.linspace(0.5, 2, 124))
        net.running_mean_std.count.fill_(7864321)
        net.value_mean_std.running_mean.fill_(158)
        net.value_mean_std.running_var.fill_(22514)
        net.value_mean_std.count.fill_(3145729)
    return net, {
        "model": net.state_dict(),
        "epoch": 192,
        "frame": 1572864,
        "optimizer": {"DO_NOT_LOAD": True},
        "env_state": {"DO_NOT_LOAD": True},
        "last_mean_rewards": 1004.0,
    }


def test_conversion_preserves_actor_critic_and_normalization_for_arbitrary_reference():
    warm = load_module("warm_start")
    old, checkpoint = donor()
    new = model(148).eval()
    before = deepcopy(checkpoint)
    converted = warm.convert_grasp_motion_weights(checkpoint, new.state_dict())
    assert set(converted) == {"model"}
    new.load_state_dict(converted["model"], strict=True)
    assert torch.count_nonzero(new.a2c_network.actor_mlp[0].weight[:, 124:]) == 0
    assert torch.equal(new.running_mean_std.running_mean[124:], torch.zeros(24, dtype=torch.float64))
    assert torch.equal(new.running_mean_std.running_var[124:], torch.ones(24, dtype=torch.float64))
    assert new.running_mean_std.count == 7864321
    x = torch.randn(32, 124)
    with torch.inference_mode():
        expected = old({"obs": x.clone(), "is_train": False})
        for reference in (torch.randn(32, 24), torch.full((32, 24), 1e5)):
            actual = new({"obs": torch.cat((x, reference), dim=1), "is_train": False})
            for key in ("mus", "sigmas", "values"):
                torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-5)
    for key in before["model"]:
        assert torch.equal(checkpoint["model"][key], before["model"][key])


def test_observer_loads_only_policy_after_exploration_and_keeps_fresh_adam(tmp_path):
    warm = load_module("warm_start")
    exploration = load_module("exploration")
    old, checkpoint = donor()
    path = tmp_path / "stage_a.pth"
    torch.save(checkpoint, path)
    # A sibling PBT companion must never be read by policy-only initialization.
    path.with_suffix(".yaml").write_text("not valid PBT metadata")
    (tmp_path / "agent.yaml").write_text(CFG.read_text())
    (tmp_path / "env.yaml").write_text("observation_space: 124\naction_space: 16\nstate_space: 0\n")
    new = model(148)
    optimizer = torch.optim.Adam(new.parameters(), lr=3e-4)
    algo = SimpleNamespace(
        model=new,
        optimizer=optimizer,
        frame=0,
        epoch_num=0,
        last_mean_rewards=-1e9,
        normalize_input=True,
        normalize_value=True,
        normalize_rms_advantage=False,
        mixed_precision=False,
        restore=lambda *a, **k: pytest.fail("Warm-start must not restore training state"),
        get_action_values=lambda obs: new({"obs": obs, "is_train": False}),
    )
    algo.set_stats_weights = lambda weights: A2CBase.set_stats_weights(algo, weights)
    algo.set_weights = lambda weights: A2CBase.set_weights(algo, weights)
    observer = exploration.ExplorationObserver(initial_std=0.15, min_std=0.03, max_std=0.3)
    observer.after_init(algo)
    try:
        warm.WarmStartObserver(path, min_std=0.03, max_std=0.3).after_init(algo)
        assert torch.equal(new.a2c_network.sigma, old.a2c_network.sigma)
        assert algo.optimizer is optimizer and not optimizer.state
        assert (algo.frame, algo.epoch_num, algo.last_mean_rewards) == (0, 0, -1e9)
        assert optimizer.param_groups[0]["lr"] == 3e-4
        out = new({"obs": torch.randn(8, 148), "prev_actions": torch.zeros(8, 16)})
        (out["mus"].square().mean() + out["values"].square().mean()).backward()
        assert new.a2c_network.actor_mlp[0].weight.grad[:, 124:].abs().max() > 0
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in new.parameters())
    finally:
        observer.close()


@pytest.mark.parametrize("change", ["missing", "shape", "nan", "normalizer_conflict"])
def test_converter_rejects_incompatible_state_without_partial_load(change):
    warm = load_module("warm_start")
    _, checkpoint = donor()
    if change == "missing":
        del checkpoint["model"]["a2c_network.value.weight"]
    elif change == "shape":
        checkpoint["model"]["a2c_network.actor_mlp.0.weight"] = torch.zeros(256, 123)
    elif change == "nan":
        checkpoint["model"]["running_mean_std.running_mean"][0] = float("nan")
    else:
        checkpoint["running_mean_std"] = {"running_mean": torch.zeros(124, dtype=torch.float64)}
    with pytest.raises(ValueError):
        warm.convert_grasp_motion_weights(checkpoint, model(148).state_dict())


def test_compiled_prefix_and_consistent_external_normalization_are_supported():
    warm = load_module("warm_start")
    _, checkpoint = donor()
    checkpoint["running_mean_std"] = {
        key: value for key, value in (("running_mean", checkpoint["model"]["running_mean_std.running_mean"]),)
    }
    checkpoint["model"] = {"_orig_mod." + key: value for key, value in checkpoint["model"].items()}
    new = model(148)
    new.load_state_dict(warm.convert_grasp_motion_weights(checkpoint, new.state_dict())["model"], strict=True)


def test_cli_warm_start_is_distinct_from_resume_and_removed_on_exploit():
    runtime = load_module("runtime")
    task = "Isaac-Grasp-Motion-Cube-LEAP-Direct-v0"
    args = SimpleNamespace(
        warm_start="stage_a.pth", checkpoint=None, pbt_resume=None, legacy_best_checkpoint=None, sigma=None, task=task
    )
    runtime.validate_warm_start_options(args)
    for key in ("checkpoint", "pbt_resume", "legacy_best_checkpoint", "sigma"):
        invalid = SimpleNamespace(**vars(args))
        setattr(invalid, key, 0.1 if key == "sigma" else "file")
        with pytest.raises(ValueError, match="warm-start"):
            runtime.validate_warm_start_options(invalid)
    args.task = "Isaac-Grasp-Cube-LEAP-Direct-v0"
    with pytest.raises(ValueError, match="warm-start"):
        runtime.validate_warm_start_options(args)
    for warm_args in (["--warm-start", "stage_a.pth"], ["--warm-start=stage_a.pth"]):
        argv = runtime.restart_command(["train.py", *warm_args, "--task", task], "python", "stage_b.pth", "b.yaml", {})
        assert not any("warm-start" in arg or "stage_a.pth" in arg for arg in argv)
        assert "--checkpoint=stage_b.pth" in argv and task in argv


def test_warm_start_requires_compatible_recipient_and_empty_best_history(tmp_path):
    runtime = load_module("runtime")
    params = yaml.safe_load(CFG.read_text())["params"]
    runtime.validate_warm_start_config(params)
    for key, value in (("activation", "relu"), ("d2rl", True), ("units", [128, 128, 64])):
        changed = deepcopy(params)
        changed["network"]["mlp"][key] = value
        with pytest.raises(ValueError, match="warm-start"):
            runtime.validate_warm_start_config(changed)
    for key, value in (("value_activation", "tanh"), ("normalization", "layer_norm"), ("central_value", True)):
        changed = deepcopy(params)
        changed["network"][key] = value
        with pytest.raises(ValueError, match="warm-start"):
            runtime.validate_warm_start_config(changed)
    runtime.validate_warm_start_destination(tmp_path / "history", tmp_path / "experiment")
    (tmp_path / "history").mkdir()
    (tmp_path / "history/best_history.json").write_text("old reward domain")
    with pytest.raises(ValueError, match="fresh"):
        runtime.validate_warm_start_destination(tmp_path / "history", tmp_path / "experiment")


def test_warm_start_requires_source_architecture_and_layout_manifest(tmp_path):
    warm = load_module("warm_start")
    path = tmp_path / "policy.pth"
    torch.save({}, path)
    with pytest.raises(ValueError, match="manifest"):
        warm.validate_grasp_source(path)
    (tmp_path / "agent.yaml").write_text(CFG.read_text())
    (tmp_path / "env.yaml").write_text("observation_space: 124\naction_space: 16\nstate_space: 0\n")
    warm.validate_grasp_source(path)
    config = yaml.safe_load(CFG.read_text())
    config["params"]["network"]["mlp"]["activation"] = "relu"
    (tmp_path / "agent.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="warm-start"):
        warm.validate_grasp_source(path)


@pytest.mark.parametrize(
    "resume_fields",
    [
        {"load_checkpoint": True, "load_path": "/old/stage_a_resume.pth"},
        {"load_checkpoint": True, "load_path": ""},
        {"load_checkpoint": False, "load_path": "/old/stage_a_resume.pth"},
    ],
)
def test_resumed_stage_a_manifest_is_valid_donor_but_not_fresh_recipient(tmp_path, resume_fields):
    warm = load_module("warm_start")
    runtime = load_module("runtime")
    manifest = yaml.safe_load(CFG.read_text())
    manifest["params"].update(resume_fields)
    (tmp_path / "agent.yaml").write_text(yaml.safe_dump(manifest))
    (tmp_path / "env.yaml").write_text("observation_space: 124\naction_space: 16\nstate_space: 0\n")
    # Donor history is provenance only; it does not ask the fresh recipient to resume.
    warm.validate_grasp_source(tmp_path / "policy.pth")
    with pytest.raises(ValueError, match="warm-start"):
        runtime.validate_warm_start_config(manifest["params"])
    manifest["params"]["network"]["mlp"]["activation"] = "relu"
    (tmp_path / "agent.yaml").write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="architecture"):
        warm.validate_grasp_source(tmp_path / "policy.pth")


def test_entrypoint_exposes_warm_start_before_launch_and_orders_it_after_exploration():
    # Entry scripts cannot be imported without starting Kit. Check their wiring,
    # while behavior is exercised through the real converter/observer above.
    source = (ROOT / "scripts/rl_games/train_dexpbt_member.py").read_text()
    assert 'parser.add_argument("--warm-start"' in source
    assert source.index("runtime.validate_warm_start_options(args_cli)") < source.index("AppLauncher(args_cli)")
    assert source.index("runtime.validate_warm_start_destination(") < source.index("env = gym.make(")
    assert source.index("exploration = ExplorationObserver(") < source.index("WarmStartObserver(")
    assert "if warm_start_path:" in source
