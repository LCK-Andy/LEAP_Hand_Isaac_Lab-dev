"""Simulator-free regression checks for the probe's reversible timing instrumentation."""

import ast
import functools
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def timers_type():
    path = Path(__file__).resolve().parents[1] / "scripts/probes/dexpbt_step_profile.py"
    tree = ast.parse(path.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ComponentTimers")
    # Execute the actual class, not an imitation; importing the CLI would start Kit.
    namespace = {"functools": functools, "time": time}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ComponentTimers"]


def test_instance_wrapper_preserves_result_and_restores_inherited_method(timers_type):
    class Target:
        def operation(self, value):
            return value + 1

    obj = Target()
    timers = timers_type()
    timers.wrap(obj, "operation", "call")
    assert obj.operation(4) == 5
    assert timers.stats["call"]["calls"] == 1
    timers.restore()
    timers.restore()  # cleanup is safe after normal restoration
    assert "operation" not in vars(obj)
    assert obj.operation(4) == 5


def test_classmethod_descriptor_restored_and_other_instances_unaffected_afterward(timers_type):
    class Target:
        @classmethod
        def operation(cls):
            return cls.__name__

    descriptor = vars(Target)["operation"]
    timers = timers_type()
    timers.wrap(Target, "operation", "class")
    assert Target.operation() == "Target"
    timers.restore()
    assert vars(Target)["operation"] is descriptor
    assert Target().operation() == "Target"


def test_exception_is_propagated_and_nested_exclusive_times_are_consistent(timers_type):
    obj = SimpleNamespace()
    obj.inner = lambda: (_ for _ in ()).throw(ValueError("fixture"))
    obj.outer = lambda: obj.inner()
    originals = dict(vars(obj))
    timers = timers_type()
    timers.wrap(obj, "inner", "inner")
    timers.wrap(obj, "outer", "outer")
    try:
        with pytest.raises(ValueError, match="fixture"):
            obj.outer()
    finally:
        timers.restore()
    assert vars(obj) == originals
    assert not timers.stack
    assert timers.stats["outer"]["exclusive_s"] >= 0
    assert timers.stats["outer"]["inclusive_s"] == pytest.approx(
        timers.stats["outer"]["exclusive_s"] + timers.stats["inner"]["inclusive_s"]
    )


def test_reset_acceptance_counts_stay_within_small_scene():
    path = Path(__file__).resolve().parents[1] / "scripts/probes/dexpbt_step_profile.py"
    tree = ast.parse(path.read_text())
    definition = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "reset_acceptance_counts"
    )
    namespace = {}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    counts = namespace["reset_acceptance_counts"]
    assert counts(1) == [1]
    assert counts(4) == [1, 4]
    assert counts(8) == [1, 4, 8]
    assert counts(32) == [1, 4, 32]
    assert counts(512) == [1, 4, 32, 512]


def test_existing_instance_override_is_restored_exactly(timers_type):
    def original(x):
        return x * 2

    obj = SimpleNamespace(operation=original)
    timers = timers_type()
    timers.wrap(obj, "operation", "call")
    assert obj.operation(3) == 6
    timers.restore()
    assert obj.operation is original
