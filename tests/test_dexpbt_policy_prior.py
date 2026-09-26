"""CPU-only real RL-Games coverage for same-task policy priors and restart isolation."""

import random
import subprocess
import sys
import torch
import yaml
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from rl_games.algos_torch.model_builder import ModelBuilder
from test_dexpbt_pbt import load_module

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt/agents/rl_games_ppo_pbt.yaml"


def agent():
    return yaml.safe_load(CFG.read_text())


def environment(num_envs=512):
    prior = load_module("policy_prior")
    return {
        "seed": 45,
        "sim": {"dt": 1 / 120, "device": "cuda:0", "gravity": (0.0, 0.0, -9.81)},
        "scene": {
            "num_envs": num_envs,
            "robot": {"spawn": {"usd_path": "kuka.usd", "spawn_path": "/World/envs/env_0/Robot"}},
            "object": {"spawn": {"spawn_paths": [f"/World/envs/env_{i}/Object" for i in range(16)]}},
        },
        "observations": {
            "policy": {"history_length": 5, "object_quat": {"func": "quat"}, "actions": {"func": "actions"}},
            "proprio": {"history_length": 5},
            "perception": {"history_length": 5, "points": {"params": {"num_points": 64}}},
        },
        "actions": {"action": {"scale": 0.1, "joint_names": [".*"]}},
        "curriculum": {"adr": {"func": prior._ADR, "params": {"init_difficulty": 0}}},
        "episode_length_s": 6.0,
    }


def model():
    return (
        ModelBuilder()
        .load(agent()["params"])
        .build(
            dict(
                actions_num=23,
                input_shape=(1745,),
                num_seqs=1,
                value_size=1,
                normalize_input=True,
                normalize_value=True,
            )
        )
    )


def save_donor(root):
    root.mkdir(parents=True, exist_ok=True)
    net = model().eval()
    with torch.no_grad():
        net.a2c_network.sigma.copy_(torch.linspace(-2.7, -0.05, 23))
        net.running_mean_std.running_mean.copy_(torch.linspace(-1, 1, 1745))
        net.running_mean_std.running_var.copy_(torch.linspace(0.5, 2, 1745))
        net.running_mean_std.count.fill_(999999)
        net.value_mean_std.running_mean.fill_(14)
        net.value_mean_std.running_var.fill_(91)
        net.value_mean_std.count.fill_(777777)
    checkpoint = {
        "model": net.state_dict(),
        "epoch": 982,
        "frame": 16089088,
        "optimizer": {"DO_NOT_LOAD": True},
        "scaler": {"DO_NOT_LOAD": True},
        "env_state": {"num_envs": 512, "curriculum": "DO_NOT_LOAD"},
        "last_mean_rewards": 777.0,
        "rng_state": random.getstate(),
    }
    path = root / "donor.pth"
    torch.save(checkpoint, path)
    (root / "agent.yaml").write_text(yaml.safe_dump(agent(), sort_keys=False))
    (root / "env.yaml").write_text(yaml.dump(environment(), sort_keys=False))
    path.with_suffix(".yaml").write_text("companion must never be read: [invalid YAML")
    return net, checkpoint, path


def fresh_algo(tmp_path):
    net = model()
    config = deepcopy(agent()["params"]["config"])
    config.update(device="cpu", num_actors=128)
    return SimpleNamespace(
        model=net,
        optimizer=torch.optim.Adam(net.parameters(), lr=3e-4),
        frame=0,
        epoch_num=0,
        last_mean_rewards=-1e9,
        config=config,
        nn_dir=str(tmp_path / "experiment" / "nn"),
        train_dir=str(tmp_path),
        game_rewards=SimpleNamespace(current_size=0),
        game_shaped_rewards=SimpleNamespace(current_size=0),
        game_lengths=SimpleNamespace(current_size=0),
        restore=lambda *args: pytest.fail("Prior must not call restore"),
        save=lambda *args: pytest.fail("Prior must not save inherited best"),
        get_action_values=lambda obs: net({"obs": obs, "is_train": False}),
    )


def test_exact_real_network_prior_keeps_fresh_optimizer_rng_curriculum_and_best(tmp_path):
    prior, exploration = load_module("policy_prior"), load_module("exploration")
    old, checkpoint, path = save_donor(tmp_path / "source")
    new = fresh_algo(tmp_path / "recipient")
    native_env = environment(128)
    config = agent()
    config["params"]["config"]["num_actors"] = 128
    bounded = exploration.ExplorationObserver(initial_std=0.15)
    bounded.after_init(new)
    before_rng, before_python = torch.get_rng_state().clone(), random.getstate()
    try:
        prior.PolicyPriorObserver(path, config, native_env, bounded).after_init(new)
        assert torch.equal(before_rng, torch.get_rng_state()) and before_python == random.getstate()
        assert (new.frame, new.epoch_num, new.last_mean_rewards) == (0, 0, -1e9)
        assert not new.optimizer.state and new.optimizer.param_groups[0]["lr"] == 3e-4
        assert native_env == environment(128) and not hasattr(new, "env_state")
        for key, value in new.model.state_dict().items():
            assert torch.equal(value, checkpoint["model"][key])
        best = load_module("best_checkpoint").BestCheckpointObserver(history_dir=tmp_path / "recipient" / "best")
        best.after_init(new)
        assert best.record["historical_best"] is None
        assert best.record["branches"]["0"]["best"] is None
        config["pbt"].update(enabled=False, policy_idx=2)
        args = SimpleNamespace(original_argv=["train.py", "--policy-prior", str(path)], checkpoint=None)
        pbt = load_module("pbt").PbtAlgoObserver(config, args)
        pbt.after_init(new)
        assert pbt.resume is None and pbt.last_exploit_frame == 0 and pbt.evidence == {}
        assert pbt.rng.getstate() == load_module("pbt_utils").recipient_rng(config["params"]["seed"], 2).getstate()
        new.model.eval()
        x = torch.randn(128, 1745)
        with torch.inference_mode():
            expected = old({"obs": x.clone(), "is_train": False})
            actual = new.get_action_values(x.clone())
            for key in ("mus", "values", "sigmas"):
                assert torch.equal(actual[key], expected[key])
        assert torch.equal(new.model.a2c_network.sigma, old.a2c_network.sigma)
        assert bounded.metrics()["exploration/projected_coordinates_total"] == 0
        new.model.train()
        output = new.model({"obs": x, "prev_actions": torch.zeros(128, 23)})
        (output["mus"].square().mean() + output["values"].square().mean()).backward()
        new.optimizer.step()
        assert new.optimizer.state and all(torch.isfinite(p).all() for p in new.model.parameters())
    finally:
        bounded.close()


@pytest.mark.parametrize(
    "compiled_source,compiled_target", [(False, False), (True, False), (False, True), (True, True)]
)
def test_strict_weights_normalize_compiled_prefixes_and_external_rms(tmp_path, compiled_source, compiled_target):
    prior = load_module("policy_prior")
    _, checkpoint, _ = save_donor(tmp_path)
    target = model().state_dict()
    checkpoint["running_mean_std"] = {"running_mean": checkpoint["model"]["running_mean_std.running_mean"]}
    checkpoint["reward_mean_std"] = {"running_var": checkpoint["model"]["value_mean_std.running_var"]}
    if compiled_source:
        checkpoint["model"] = {"_orig_mod." + key: value for key, value in checkpoint["model"].items()}
    if compiled_target:
        target = {"_orig_mod." + key: value for key, value in target.items()}
    converted = prior.model_weights(checkpoint, target)
    assert converted.keys() == target.keys()
    assert all(
        torch.equal(
            value, checkpoint["model"][("_orig_mod." if compiled_source else "") + key.removeprefix("_orig_mod.")]
        )
        for key, value in converted.items()
    )


@pytest.mark.parametrize("invalid", ["missing", "shape", "dtype", "nan", "variance", "count", "external", "duplicate"])
def test_weights_fail_closed_without_partial_load(tmp_path, invalid):
    prior = load_module("policy_prior")
    _, checkpoint, _ = save_donor(tmp_path)
    state = checkpoint["model"]
    if invalid == "missing":
        del state["a2c_network.critic_mlp.0.weight"]
    elif invalid == "shape":
        state["a2c_network.actor_mlp.0.weight"] = torch.zeros(512, 1744)
    elif invalid == "dtype":
        state["a2c_network.sigma"] = state["a2c_network.sigma"].double()
    elif invalid == "nan":
        state["a2c_network.sigma"][0] = float("nan")
    elif invalid == "variance":
        state["running_mean_std.running_var"][0] = -1
    elif invalid == "count":
        state["value_mean_std.count"].zero_()
    elif invalid == "external":
        checkpoint["reward_mean_std"] = {"running_mean": torch.zeros(1)}
    else:
        state["_orig_mod.a2c_network.sigma"] = state["a2c_network.sigma"]
    new = model()
    before = deepcopy(new.state_dict())
    with pytest.raises(ValueError):
        prior.model_weights(checkpoint, new.state_dict())
    assert all(torch.equal(value, before[key]) for key, value in new.state_dict().items())


@pytest.mark.parametrize("field", ["frame", "epoch_num", "optimizer", "last_mean_rewards", "game_rewards", "sigma"])
def test_observer_refuses_nonfresh_state_or_nonexact_sigma(tmp_path, field):
    prior, exploration = load_module("policy_prior"), load_module("exploration")
    _, checkpoint, path = save_donor(tmp_path)
    new = fresh_algo(tmp_path)
    if field == "optimizer":
        new.optimizer.state["old"] = {}
    elif field == "game_rewards":
        new.game_rewards.current_size = 1
    elif field != "sigma":
        setattr(new, field, 1)
    else:
        checkpoint["model"]["a2c_network.sigma"][0] = 0.01
        torch.save(checkpoint, path)
    bounded = exploration.ExplorationObserver()
    bounded.after_init(new)
    before = deepcopy(new.model.state_dict())
    try:
        with pytest.raises(ValueError):
            prior.PolicyPriorObserver(path, agent(), environment(128), bounded).after_init(new)
        assert all(torch.equal(value, before[key]) for key, value in new.model.state_dict().items())
    finally:
        bounded.close()


def test_exploration_must_initialize_first_and_arm_bounds_use_actual_mapping(tmp_path):
    prior, exploration = load_module("policy_prior"), load_module("exploration")
    _, _, path = save_donor(tmp_path)
    new = fresh_algo(tmp_path)
    names = [f"iiwa7_joint_{i}" for i in range(1, 8)] + [
        f"{finger}_joint_{i}" for finger in ("index", "middle", "ring", "thumb") for i in range(4)
    ]
    bounded = exploration.ExplorationObserver(action_joint_names=list(reversed(names)))
    observer = prior.PolicyPriorObserver(path, agent(), environment(128), bounded)
    with pytest.raises(ValueError, match="after the shared exploration"):
        observer.after_init(new)
    bounded.after_init(new)
    try:
        # The high-std donor tail belongs to the arm in this reversed verified order.
        with pytest.raises(ValueError, match="sigma exceeds"):
            observer.after_init(new)
    finally:
        bounded.close()


@pytest.mark.parametrize("template", ["/World/envs/env_{}", "/World/custom/cell_{}"])
def test_pre_gym_namespaces_match_post_gym_manifest_without_loosening_regex(tmp_path, template):
    prior = load_module("policy_prior")
    _, _, path = save_donor(tmp_path)
    pre = environment(128)
    pre["scene"]["clone_cfg"] = {"clone_template": template}
    pre["scene"]["robot"]["prim_path"] = "{ENV_REGEX_NS}/Robot"
    pre["scene"]["object"]["prim_path"] = "{ENV_REGEX_NS}/Object"
    pre["scene"]["sensor"] = {
        "prim_path": "{ENV_REGEX_NS}/Robot/ee_link/index_link_3",
        "filter_prim_paths_expr": ["{ENV_REGEX_NS}/Object"],
    }
    post = deepcopy(pre)
    post["scene"]["num_envs"] = 512
    for name in ("robot", "object", "sensor"):
        post["scene"][name]["prim_path"] = pre["scene"][name]["prim_path"].format(ENV_REGEX_NS=template.format("[^/]+"))
    post["scene"]["sensor"]["filter_prim_paths_expr"] = [template.format("[^/]+") + "/Object"]
    (path.parent / "env.yaml").write_text(yaml.dump(post, sort_keys=False))
    prior.validate_source(path, agent(), pre)
    post["scene"]["sensor"]["filter_prim_paths_expr"] = [template.format(".*") + "/Object"]
    (path.parent / "env.yaml").write_text(yaml.dump(post, sort_keys=False))
    with pytest.raises(ValueError, match="filter_prim_paths_expr"):
        prior.validate_source(path, agent(), pre)


def test_registered_native_pre_gym_cfg_matches_real_saved_post_gym_manifest(tmp_path):
    # Normal task registration installs shared reward wrappers. Isolate it from
    # unit-test module stubs, never bypass package registration with spec imports.
    run = ROOT / "logs/rl_games/dexpbt_download_weste_20260926_230552_member03"
    if not (run / "params/env.yaml").is_file():
        pytest.skip("Optional local real donor manifest is unavailable")
    script = """
import sys
from pathlib import Path
import yaml
import isaaclab_tasks
import LEAP_Isaaclab.tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt.policy_prior import TASK, validate_source
cfg = parse_env_cfg(TASK, device='cuda:0', num_envs=128)
assert cfg.scene.robot.prim_path == '{ENV_REGEX_NS}/Robot'
run = Path(sys.argv[1])
params = yaml.safe_load(Path(sys.argv[2]).read_text())
validate_source(run / 'nn/000000000032.pth', params, cfg.to_dict())
print('native-pre-gym-to-real-post-gym: PASS')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(run), str(CFG)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "native-pre-gym-to-real-post-gym: PASS" in completed.stdout


def test_manifest_layouts_numeric_equivalence_and_512_to_128(tmp_path):
    prior = load_module("policy_prior")
    _, _, path = save_donor(tmp_path / "run")
    env = environment(128)
    env["seed"] = 999
    env["sim"]["device"] = "cpu"
    env["scene"]["object"]["spawn"]["spawn_paths"] = [None] * 16
    env["curriculum"]["adr"]["params"]["init_difficulty"] = -0.0
    donor_agent = agent()
    donor_agent["params"].update(load_checkpoint=True, load_path="old-resume.pth")
    (path.parent / "agent.yaml").write_text(yaml.safe_dump(donor_agent, sort_keys=False))
    assert prior.validate_source(path, agent(), env) == path.parent
    params = path.parent / "params"
    params.mkdir()
    for name in ("agent.yaml", "env.yaml"):
        (path.parent / name).rename(params / name)
    assert prior.validate_source(path, agent(), env) == params
    nn = path.parent / "nn"
    nn.mkdir()
    relocated = nn / path.name
    path.rename(relocated)
    assert prior.validate_source(relocated, agent(), env) == params
    for name in ("agent.yaml", "env.yaml"):
        (nn / name).write_text((params / name).read_text())
    with pytest.raises(ValueError, match="unambiguous"):
        prior.validate_source(relocated, agent(), env)


@pytest.mark.parametrize(
    "invalid", ["task", "observation_order", "history", "action", "physics", "episode", "curriculum"]
)
def test_manifests_reject_different_mdp(tmp_path, invalid):
    prior = load_module("policy_prior")
    _, _, path = save_donor(tmp_path)
    env = environment(128)
    if invalid == "task":
        env["curriculum"]["adr"]["func"] = "not-the-PBT-task"
    elif invalid == "observation_order":
        env["observations"]["policy"] = dict(reversed(list(env["observations"]["policy"].items())))
    elif invalid == "history":
        env["observations"]["proprio"]["history_length"] = 4
    elif invalid == "action":
        env["actions"]["action"]["scale"] = 0.2
    elif invalid == "physics":
        env["sim"]["dt"] = 1 / 60
    elif invalid == "episode":
        env["episode_length_s"] = 10
    else:
        env["curriculum"]["adr"]["params"]["init_difficulty"] = 1
    with pytest.raises(ValueError, match="policy-prior"):
        prior.validate_source(path, agent(), env)


@pytest.mark.parametrize(
    "change", ["activation", "separate", "groups", "normalize_input", "normalize_value", "rnn", "load_path"]
)
def test_agent_architecture_and_freshness_are_strict(change):
    prior = load_module("policy_prior")
    cfg = agent()
    if change == "activation":
        cfg["params"]["network"]["mlp"]["activation"] = "relu"
    elif change == "separate":
        cfg["params"]["network"]["separate"] = False
    elif change == "groups":
        cfg["params"]["env"]["obs_groups"].reverse()
    elif change.startswith("normalize"):
        cfg["params"]["config"][change] = False
    elif change == "rnn":
        cfg["params"]["network"]["rnn"] = {}
    else:
        cfg["params"]["load_path"] = "old.pth"
    with pytest.raises(ValueError, match="policy-prior"):
        prior.validate_architecture(cfg, recipient=True)


def test_cli_rejects_every_resume_path_and_exploit_drops_prior(tmp_path):
    runtime = load_module("runtime")
    args = SimpleNamespace(policy_prior="donor.pth", task=load_module("policy_prior").TASK)
    runtime.validate_policy_prior_options(args)
    for field in ("checkpoint", "pbt_resume", "warm_start", "legacy_best_checkpoint", "sigma"):
        changed = SimpleNamespace(**vars(args), **{field: 0 if field == "sigma" else "old.pth"})
        with pytest.raises(ValueError, match="policy-prior"):
            runtime.validate_policy_prior_options(changed)
    for key, value in (("params.load_checkpoint", False), ("agent.params.load_path", "")):
        with pytest.raises(ValueError, match="policy-prior"):
            runtime.validate_policy_prior_options(args, {key: value})
    args.task = "Isaac-Grasp-Motion-Cube-LEAP-Direct-v0"
    with pytest.raises(ValueError):
        runtime.validate_policy_prior_options(args)
    for flags in (["--policy-prior", "donor.pth"], ["--policy-prior=donor.pth"]):
        argv = runtime.restart_command(["train.py", *flags, "--num_envs=128"], "python", "new.pth", "new.yaml", {})
        assert argv == ["python", "train.py", "--num_envs=128", "--checkpoint=new.pth", "--pbt-resume=new.yaml"]
    runtime.validate_policy_prior_destination(tmp_path / "member", tmp_path / "experiment")
    (tmp_path / "member").mkdir()
    (tmp_path / "member" / "old_score.yaml").write_text("old")
    with pytest.raises(ValueError, match="fresh"):
        runtime.validate_policy_prior_destination(tmp_path / "member", tmp_path / "experiment")


@pytest.mark.parametrize("destination", ["member", "experiment"])
@pytest.mark.parametrize("existing", ["best_history", "checkpoint", "symlink"])
def test_nonfresh_destinations_refused(tmp_path, destination, existing):
    runtime = load_module("runtime")
    path = tmp_path / destination
    if existing == "symlink":
        path.symlink_to(tmp_path / "missing", target_is_directory=True)
    else:
        path.mkdir()
        (path / ("best_history.json" if existing == "best_history" else "old.pth")).write_text("old")
    with pytest.raises(ValueError, match="fresh"):
        runtime.validate_policy_prior_destination(tmp_path / "member", tmp_path / "experiment")


def test_live_group_dimensions_are_checked():
    prior = load_module("policy_prior")
    env = SimpleNamespace(
        observation_manager=SimpleNamespace(group_obs_dim=deepcopy(prior.GROUP_DIMS)),
        action_manager=SimpleNamespace(total_action_dim=23),
    )
    prior.validate_live_layout(env)
    env.observation_manager.group_obs_dim["policy"] = (171,)
    with pytest.raises(ValueError):
        prior.validate_live_layout(env)


def test_entrypoint_orders_prior_after_exploration_before_best_and_has_prelaunch_guard():
    source = (ROOT / "scripts/rl_games/train_dexpbt_member.py").read_text()
    assert '"--policy-prior", type=str' in source
    assert source.index("runtime.validate_policy_prior_options(") < source.index("AppLauncher(args_cli)")
    assert source.index("runtime.validate_policy_prior_destination(") < source.index("env = gym.make(")
    assert (
        source.index("exploration = ExplorationObserver(")
        < source.index("        append_policy_prior(")
        < source.index("observers.append(BestCheckpointObserver(")
    )
