"""CPU-only public RL-Games execution and isolated resume contract tests."""

import copy
import sys
import yaml
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Standalone test invocation requires the repository path bootstrap before local imports.
from scripts.dexpbt_original.pbt import OriginalPbtObserver, PbtConfig, restart_state  # noqa: E402
from scripts.dexpbt_original.runtime import (  # noqa: E402
    ENV_DEFAULTS,
    PPO_PREFIX,
    apply_mutable,
    budget_reached,
    compatibility_stamp,
    mutable_values,
    restart_command,
    restore_agent,
    validate_checkpoint,
    validate_environment,
)

AGENT = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexpbt_original/agents/rl_games_ppo.yaml"


def agent_config():
    return yaml.safe_load(AGENT.read_text())


def test_imports_do_not_register_simulator_tasks():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scripts.dexpbt_original.train; import sys; assert 'pxr' not in sys.modules; assert 'LEAP_Isaaclab' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_play_parser_coexists_with_native_launcher():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.dexpbt_original.play import build_parser; from isaaclab.app import AppLauncher; p=build_parser(); AppLauncher.add_app_launcher_args(p); a=p.parse_args(['--checkpoint=x.pth','--deterministic-policy']); assert a.deterministic_policy; assert hasattr(a,'deterministic')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "script,arguments",
    [("train.py", ["--workspace=/tmp/not-created"]), ("play.py", ["--checkpoint=/tmp/not-read.pth"])],
)
def test_physics_mismatch_guard_refuses_before_simulator_import(script, arguments):
    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/dexpbt_original" / script), *arguments],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "--allow-physics-mismatch" in result.stderr and "signed inertia" in result.stderr
    assert "Starting Isaac" not in result.stdout


def test_manual_resume_restarts_original_adaptation_clock(tmp_path):
    import random

    resume = {
        "resume_frame": 300_000_000,
        "segment_start_frame": 200_000_000,
        "rng_state": random.Random(42).getstate(),
    }
    observer = OriginalPbtObserver(
        PbtConfig(), tmp_path, mutable_values(agent_config(), ENV_DEFAULTS), "same", 2, 42, ["train.py"], resume
    )
    assert observer.segment_start == 300_000_000
    assert 320_000_000 - observer.segment_start < observer.cfg.adaptation


def test_mutations_must_be_applied_before_constructor_and_read_back():
    agent = agent_config()
    cfg = SimpleNamespace(original_mutable_params={})
    values = mutable_values(agent, ENV_DEFAULTS)
    values["task.env.liftingBonus"] = 500.0
    values[PPO_PREFIX + "mini_epochs"] = 3
    apply_mutable(agent, cfg, values)
    assert cfg.original_mutable_params["liftingBonus"] == 500.0
    assert agent["params"]["config"]["mini_epochs"] == 3
    env = SimpleNamespace(
        get_original_mutable_params=lambda: cfg.original_mutable_params,
        get_pbt_env_state=lambda: {"success_tolerance": 0.05},
        set_pbt_env_state=lambda state: None,
    )
    validate_environment(env, values)
    cfg.original_mutable_params["liftingBonus"] = 300.0
    with pytest.raises(ValueError, match="readback"):
        validate_environment(env, values)
    with pytest.raises(ValueError, match="exactly"):
        mutable_values(agent, {**ENV_DEFAULTS, "distRewardScale": 1.0})


def test_restart_cli_retains_member_device_and_budget():
    argv = [
        "/tmp/train.py",
        "--member",
        "3",
        "--device=cuda:1",
        "--checkpoint",
        "/old path/file.pth",
        "--max-frames",
        "500",
        "--mini-epochs=1",
        "--minibatch-size",
        "64",
    ]
    result = restart_command(argv, "/new path/file.pth")
    assert result == [
        "/tmp/train.py",
        "--member",
        "3",
        "--device=cuda:1",
        "--max-frames",
        "500",
        "--checkpoint=/new path/file.pth",
    ]


def test_config_package_has_file_for_registry_loader():
    import importlib.util

    init = AGENT.parent / "__init__.py"
    spec = importlib.util.spec_from_file_location("original_agents_fixture", init)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = Path(module.__file__).parent / "rl_games_ppo.yaml"
    assert yaml.safe_load(path.read_text())["params"]["network"]["rnn"]["units"] == 768


def test_launch_generator_is_explicit_dry_run(tmp_path):
    from scripts.dexpbt_original.launch import commands

    result = commands(tmp_path, [str(i) for i in range(8)])
    assert len(result) == 8
    assert all("--allow-physics-mismatch" not in command for command in result)
    opted = commands(tmp_path, [str(i) for i in range(8)], allow_physics_mismatch=True)
    assert all("--allow-physics-mismatch" in command for command in opted)
    assert all("--max-frames=5000000000" in command for command in result)
    assert "--device=cuda:7" in result[7] and "--member=7" in result[7]
    with pytest.raises(ValueError, match="unique"):
        commands(tmp_path, ["0"] * 8)
    (tmp_path / "007").mkdir()
    (tmp_path / "007" / "final.pth").touch()
    with pytest.raises(ValueError, match="populated"):
        commands(tmp_path, [str(i) for i in range(8)])


def test_stamp_rejects_fixed_ppo_changes_not_mutations():
    agent = agent_config()
    first = compatibility_stamp(agent, {}, "released_source", 8192)
    agent["params"]["config"]["gamma"] = 0.98
    assert first == compatibility_stamp(agent, {}, "released_source", 8192)
    agent["params"]["config"]["tau"] = 0.5
    assert first != compatibility_stamp(agent, {}, "released_source", 8192)
    agent = agent_config()
    agent["params"]["config"]["minibatch_size"] = 16384
    assert first != compatibility_stamp(agent, {}, "released_source", 8192)


def test_stamp_hashes_actual_asset_bytes(tmp_path):
    robot = tmp_path / "robot.usd"
    manifest = tmp_path / "manifest.json"
    robot.write_text("first")
    manifest.write_text("{}")
    env = {"robot_usd_path": str(robot)}
    first = compatibility_stamp(agent_config(), env, "released_source", 8192)
    robot.write_text("modified")
    assert first != compatibility_stamp(agent_config(), env, "released_source", 8192)


def test_stamp_ignores_reward_mutations_seed_device_but_not_task_or_layout():
    agent = agent_config()
    env = {
        "seed": 42,
        "sim": {"device": "cuda:0", "dt": 0.01667},
        "original_mutable_params": ENV_DEFAULTS,
        "observation_space": 117,
    }
    first = compatibility_stamp(agent, env, "released_source", 8192)
    other = copy.deepcopy(env)
    other["seed"] = 43
    other["sim"]["device"] = "cuda:1"
    other["original_mutable_params"]["liftingBonus"] = 600
    assert first == compatibility_stamp(agent, other, "released_source", 8192)
    assert first != compatibility_stamp(agent, other, "paper", 8192)
    assert first != compatibility_stamp(agent, other, "released_source", 1024)
    other["sim"]["dt"] = 0.02
    assert first != compatibility_stamp(agent, other, "released_source", 8192)


def test_real_process_original_observer_reexec(tmp_path):
    """Execute the real os.execv path; pair with native PPO state test below."""
    import os
    import subprocess
    import textwrap

    script = tmp_path / "member.py"
    script.write_text(textwrap.dedent("""
        import argparse, copy, json, sys, time
        from pathlib import Path
        from types import SimpleNamespace
        import torch, yaml
        from scripts.dexpbt_original.pbt import OriginalPbtObserver, PbtConfig, atomic_checkpoint, atomic_json, population_contract
        from scripts.dexpbt_original.runtime import ENV_DEFAULTS, mutable_values, validate_checkpoint
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint')
        parser.add_argument('--member', type=int, required=True)
        parser.add_argument('--minibatch-size')
        parser.add_argument('--mini-epochs')
        args = parser.parse_args()
        root = Path(__file__).parent
        if args.checkpoint:
            assert args.minibatch_size is None and args.mini_epochs is None
            saved = torch.load(args.checkpoint, weights_only=False)
            meta = validate_checkpoint(saved, 0, 'released_source')
            assert saved['frame'] == 16 and saved['epoch'] == 8
            assert meta['lineage'][-1]['donor'] in (2, 3)
            assert meta['segment_start_frame'] == 16
            assert meta['mutation_count'] == 1
            assert saved['optimizer']['state']
            (root / 'success.json').write_text(json.dumps({'member': meta['member'], 'frame': saved['frame']}))
            sys.exit(0)
        config_path = Path(__import__('scripts.dexpbt_original.runtime', fromlist=['ROOT']).__file__).parents[2] / 'source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexpbt_original/agents/rl_games_ppo.yaml'
        agent = yaml.safe_load(config_path.read_text())
        values = mutable_values(agent, ENV_DEFAULTS)
        cfg = PbtConfig(member=args.member, population=4, interval=16, initial_delay=0, adaptation=0, mutation_rate=1.0, minimum_runtime_seconds=0)
        observer = OriginalPbtObserver(cfg, root/'population', values, 'same', 2, 42, sys.argv, agent_config=agent)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model(torch.ones(1, 2)).sum().backward(); optimizer.step()
        raw = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'env_state': {'success_tolerance': .03}, 'frame':16, 'epoch':8}
        observer.after_init(SimpleNamespace(frame=16, epoch_num=8, max_frames=1000, max_epochs=100,
            get_full_state_weights=lambda: copy.deepcopy(raw), writer=SimpleNamespace(add_scalar=lambda *a:None, flush=lambda:None, close=lambda:None)))
        own = observer.checkpoint_state()
        if args.member:
            member = args.member
            own['frame'], own['epoch'] = 16, 1
            path = observer.directory/'000000001.pth'; atomic_checkpoint(path, own)
            atomic_json(path.with_suffix('.json'), {'member':member, 'iteration':1, 'frame':16,
                'objective':float(member), 'mutable':values, 'stamp':'same', 'profile':cfg.profile,
                'population_contract': population_contract(cfg), 'checkpoint':str(path)})
            sys.exit(0)
        deadline = time.monotonic() + 45
        while not all((root/'population'/f'{member:03d}'/'000000001.json').is_file() for member in (1,2,3)):
            if time.monotonic() > deadline:
                raise RuntimeError('Timed out waiting for independently published donor processes')
            time.sleep(.05)
        observer.window.add([0.,0.], [0,1])
        observer.after_print_stats(0,8,0)
        raise AssertionError('Expected real process replacement')
    """))
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    processes = []
    try:
        for member in range(4):
            processes.append(
                subprocess.Popen(
                    [sys.executable, str(script), f"--member={member}", "--minibatch-size=64", "--mini-epochs", "1"],
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        for process in processes:
            stdout, stderr = process.communicate(timeout=60)
            assert process.returncode == 0, stdout + stderr
        assert (tmp_path / "success.json").is_file()
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()


def test_real_cpu_recurrent_ppo_checkpoint_and_mutated_restore(tmp_path):
    torch = pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("rl_games")
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    torch.set_num_threads(1)

    class ToyEnv:
        def __init__(self):
            self.step_count = 0
            self.mutable = dict(ENV_DEFAULTS)
            self.tolerance = 0.03

        def get_env_info(self):
            return {
                "observation_space": gym.spaces.Box(-1.0, 1.0, (5,)),
                "action_space": gym.spaces.Box(-1.0, 1.0, (2,)),
                "agents": 1,
                "value_size": 1,
            }

        def get_number_of_agents(self):
            return 1

        def set_train_info(self, *args, **kwargs):
            pass

        def reset(self):
            return torch.zeros((2, 5))

        def step(self, actions):
            self.step_count += 1
            done = torch.full((2,), self.step_count % 4 == 0, dtype=torch.bool)
            return torch.zeros((2, 5)), torch.ones(2), done, {"true_objective": torch.tensor([1.0, 2.0])}

        def get_original_mutable_params(self):
            return self.mutable

        def get_pbt_env_state(self):
            return {"success_tolerance": self.tolerance}

        def set_pbt_env_state(self, state):
            self.tolerance = state["success_tolerance"]

        def get_env_state(self):
            return self.get_pbt_env_state()

        def set_env_state(self, state):
            self.set_pbt_env_state(state)

    env = ToyEnv()
    vecenv.register("OriginalCpuOracle", lambda config_name, num_actors, **kw: env)
    env_configurations.register(
        "original_cpu_oracle", {"vecenv_type": "OriginalCpuOracle", "env_creator": lambda **kw: env}
    )
    config = agent_config()
    config["params"]["config"].update(
        env_name="original_cpu_oracle",
        device="cpu",
        device_name="cpu",
        num_actors=2,
        mixed_precision=False,
        minibatch_size=32,
        max_epochs=1,
        max_frames=-1,
        train_dir=str(tmp_path),
        full_experiment_name="native",
        print_stats=False,
    )
    # Exercise the exact original LSTM/MLP widths on CPU, not an MLP stand-in.
    values = mutable_values(config, ENV_DEFAULTS)
    cfg = PbtConfig(enabled=False)
    observer = OriginalPbtObserver(
        cfg, tmp_path / "population", values, "cpu", 2, 42, ["train.py"], agent_config=config
    )
    runner = Runner(observer)
    runner.load(config)
    algo = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
    assert algo.is_rnn
    algo.train()
    assert budget_reached(algo)
    checkpoint = observer.publish(terminal=True)
    validate_checkpoint(checkpoint)
    assert checkpoint["optimizer"]["state"]
    assert checkpoint["env_state"] == {"success_tolerance": 0.03}
    assert checkpoint["frame"] == 32
    assert observer.window.current == 1.5
    # A donor restart inherits learned state, but constructs changed PPO settings
    # first and has to preserve recipient frame/epoch and narrow tolerance state.
    changed = dict(values)
    changed[PPO_PREFIX + "mini_epochs"] = 3
    changed[PPO_PREFIX + "gamma"] = 0.98
    changed[PPO_PREFIX + "kl_threshold"] = 0.02
    changed["task.env.liftingBonus"] = 450.0
    recipient = copy.deepcopy(checkpoint)
    recipient["frame"], recipient["epoch"] = 320, 10
    restarted = restart_state(checkpoint, recipient, changed, observer.rng.getstate())
    env_cfg = SimpleNamespace(original_mutable_params={})
    apply_mutable(config, env_cfg, changed)
    env.mutable = env_cfg.original_mutable_params
    env.tolerance = 0.075
    new_observer = OriginalPbtObserver(
        cfg, tmp_path / "resume", changed, "cpu", 2, 42, ["train.py"], agent_config=config
    )
    new_runner = Runner(new_observer)
    new_runner.load(config)
    restored = new_runner.algo_factory.create(new_runner.algo_name, base_name="run", params=new_runner.params)
    restore_agent(restored, restarted, env, changed)
    assert (restored.frame, restored.epoch_num) == (320, 10)
    assert restored.mini_epochs_num == 3 and restored.gamma == 0.98
    assert restored.scheduler.kl_threshold == 0.02
    assert env.mutable["liftingBonus"] == 450.0 and env.tolerance == 0.03
    assert restored.last_lr == restored.optimizer.param_groups[0]["lr"]
    for key, tensor in algo.model.state_dict().items():
        assert torch.equal(restored.model.state_dict()[key], tensor)
    algo.writer.close()
    restored.writer.close()
