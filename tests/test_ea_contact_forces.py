"""Exercise the actual tactile property without importing simulator packages."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_tactile_forces_use_only_configured_partners():
    path = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/leap_direct_env.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LeapHandDirectEnv")
    prop = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "fingertip_contact_forces")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[prop], type_ignores=[]), str(path), "exec"), scope)
    partner = torch.tensor([[[[1.0, 2.0, 3.0], [0.5, 0.0, 0.0]]]])
    data = SimpleNamespace(
        normal_force_matrix_w=SimpleNamespace(torch=partner),
        net_forces_w=SimpleNamespace(torch=torch.full((1, 1, 3), 100.0)),
    )
    env = SimpleNamespace(
        num_envs=1, device="cpu", _physics_backend="physx",
        cfg=SimpleNamespace(fingertip_body_names=["tip"]),
        fingertip_contact_sensors={"tip": SimpleNamespace(data=data)},
    )
    actual = scope["fingertip_contact_forces"].fget(env)
    torch.testing.assert_close(actual, torch.tensor([[[1.5, 2.0, 3.0]]]))
