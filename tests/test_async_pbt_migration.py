"""Application compatibility contracts against the installed public PBT engine."""

import copy
import subprocess
import sys
import torch
import types
import warnings

import pytest
from async_pbt.integrations.rl_games import pbt as public_pbt
from test_dexpbt_pbt import PBT, FakeAlgo, config, load_module


def test_runtime_import_is_lightweight_before_launcher():
    script = f"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location('runtime', {str(PBT / 'runtime.py')!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.split_overrides(['pbt.policy_idx=1'])[0] == {{'policy_idx': 1}}
assert not any(name.split('.')[0] in {{'torch', 'rl_games', 'pxr', 'isaaclab', 'isaacsim', 'LEAP_Isaaclab'}} for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_default_normalized_before_both_public_entrypoints(monkeypatch):
    local = load_module("pbt")
    settings = config()
    seen = []
    monkeypatch.setattr(
        local,
        "public_initialize_population",
        lambda params, resumed: seen.append(params["pbt"]["objective"]),
    )
    local.initialize_population(settings)
    assert seen == ["dexpbt"]
    settings = config()
    observer = local.PbtAlgoObserver(settings, types.SimpleNamespace(original_argv=["train.py"]))
    assert observer.cfg.objective == settings["pbt"]["objective"] == "dexpbt"
    assert isinstance(observer, public_pbt.PbtAlgoObserver)
    assert local.PbtAlgoObserver.after_steps is public_pbt.PbtAlgoObserver.after_steps
    assert local.MultiObserver is public_pbt.MultiObserver


@pytest.mark.parametrize("metadata", [None, {}, {"frame": 500, "epoch": 5, "recipient_frame": 700}])
def test_directory_and_restore_compatibility(tmp_path, monkeypatch, metadata):
    local = load_module("pbt")
    settings = config()
    settings["pbt"].update(directory=str(tmp_path / "launch-base"), workspace="legacy-workspace")
    original = copy.deepcopy(settings)
    argv = ["train.py", f"pbt.directory={settings['pbt']['directory']}"]
    args = types.SimpleNamespace(original_argv=argv, checkpoint="raw.pth", pbt_resume_metadata=metadata)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        observer = local.PbtAlgoObserver(settings, args)
    algo = FakeAlgo(tmp_path / "actual-train-dir", settings, "recipient")
    captured = []
    install = public_pbt.install_restore

    def capture(algo, params, companion, **kwargs):
        captured.append(companion)
        return install(algo, params, companion, **kwargs)

    monkeypatch.setattr(public_pbt, "install_restore", capture)
    observer.after_init(algo)
    assert observer.resume is metadata
    assert captured[0] == ({} if metadata is None else metadata)
    if metadata is not None:
        assert captured[0] is metadata
    else:
        assert captured[0] is not None
    assert observer.ws_dir == str(tmp_path / "actual-train-dir" / "legacy-workspace")
    assert observer.cfg.directory == original["pbt"]["directory"]
    assert settings == {**original, "pbt": {**original["pbt"], "objective": "dexpbt"}}
    assert observer.original_argv == argv
    donor = FakeAlgo(tmp_path, settings, "donor")
    donor.optimizer.param_groups[0]["lr"] = 0.009
    donor.frame, donor.epoch_num = 300, 3
    path = tmp_path / "raw.pth"
    torch.save(
        {
            **donor.get_full_state_weights(),
            "pbt_effective_params": {"agent.params.config.grad_norm": 9},
        },
        path,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        algo.restore(path)
    assert not any("preserving saved optimizer LR" in str(item.message) for item in caught)
    assert algo.optimizer.param_groups[0]["lr"] == algo.last_lr == settings["params"]["config"]["learning_rate"]
    assert algo.grad_norm == 1
    assert algo.frame == (700 if metadata else 300)
    assert observer.plain_checkpoint == (metadata is None)
    if metadata is None:
        assert observer.pbt_it == 3


def test_facades_are_public_exports():
    from async_pbt import mutation
    from async_pbt.adapters import allegro, contact, grasp, grasp_motion
    from async_pbt.integrations.rl_games import best_checkpoint, exploration

    for name, public, symbol in [
        ("mutation", mutation, "mutate"),
        ("allegro", allegro, "GoalWindow"),
        ("contact", contact, "ContactWindow"),
        ("grasp", grasp, "GraspHoldWindow"),
        ("grasp_motion", grasp_motion, "GraspMotionWindow"),
        ("best_checkpoint", best_checkpoint, "BestCheckpointObserver"),
        ("exploration", exploration, "ExplorationObserver"),
    ]:
        assert getattr(load_module(name), symbol) is getattr(public, symbol)


def test_missing_framework_has_actionable_install_error(monkeypatch):
    from importlib import metadata

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", missing)
    with pytest.raises(ImportError, match="pip install --no-deps"):
        load_module("runtime")


def test_runtime_delegates_to_installed_distribution():
    from importlib.metadata import distribution
    from pathlib import Path

    runtime = load_module("runtime")
    assert Path(runtime.install_restore.__code__.co_filename) == distribution("async-pbt").locate_file(
        "async_pbt/integrations/rl_games/runtime.py"
    )
