"""Native route prints source commands; it never launches training or alters the stack."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dexpbt_native_launch", ROOT / "scripts/dexpbt_original/launch_native.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def test_native_commands_retain_original_source_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(
        launch.subprocess, "check_output", lambda argv, **kw: launch.COMMIT if "rev-parse" in argv else ""
    )
    commands = launch.commands(tmp_path, tmp_path / "workspace", sys.executable, [str(i) for i in range(8)])
    assert len(commands) == 8
    for member, command in enumerate(commands):
        assert f"CUDA_VISIBLE_DEVICES={member}" in command
        assert "-m isaacgymenvs.train" in command
        assert "task=AllegroKukaLSTM task/env=reorientation" in command
        assert "train.params.config.max_frames=5000000000" in command
        assert "pbt.start_after=100000000" in command
        assert "pbt.initial_delay=200000000" in command
        assert "pbt.interval_steps=20000000" in command
        assert f"pbt.policy_idx={member}" in command
        assert f"seed={42 + member}" in command
        assert "wandb_activate=False" in command


def test_native_route_rejects_shared_gpu_and_populated_workspace(tmp_path):
    with pytest.raises(ValueError, match="distinct GPU"):
        launch.commands(tmp_path, tmp_path / "workspace", sys.executable, ["0"] * 8)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "checkpoint.pth").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="empty workspace"):
        launch.commands(tmp_path, workspace, sys.executable, [str(i) for i in range(8)])
    assert (workspace / "checkpoint.pth").read_bytes() == b"preserve"


def test_native_route_rejects_wrong_or_modified_source(tmp_path, monkeypatch):
    devices = [str(i) for i in range(8)]
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *args, **kwargs: "wrong")
    with pytest.raises(ValueError, match="pinned"):
        launch.commands(tmp_path, tmp_path / "workspace", sys.executable, devices)
    monkeypatch.setattr(
        launch.subprocess, "check_output", lambda argv, **kw: launch.COMMIT if "rev-parse" in argv else " M asset"
    )
    with pytest.raises(ValueError, match="clean"):
        launch.commands(tmp_path, tmp_path / "workspace", sys.executable, devices)
