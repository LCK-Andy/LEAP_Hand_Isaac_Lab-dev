"""Original released catalog generation is the oracle, including duplicate dimensions."""

import hashlib
import importlib.util
import json
import numpy as np
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/dexpbt_original"
TASK = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexpbt_original"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_upstream_oracles_are_unmodified():
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    assert manifest["commit"] == "665c32170d84b4be66722eea405a1e08b6e7f761"
    for filename, digest in manifest["files"].items():
        assert hashlib.sha256((FIXTURE / filename).read_bytes()).hexdigest() == digest, filename


@pytest.mark.parametrize(
    "randomize,small,big,sticks",
    [(True, True, True, True), (False, True, True, True), (True, True, False, False), (True, False, True, True)],
)
def test_catalog_matches_actual_released_generator(tmp_path, randomize, small, big, sticks):
    original = load_module("dexpbt_source_cuboids", FIXTURE / "isaacgymenvs/tasks/allegro_kuka/generate_cuboids.py")
    original.generate_default_cube(str(tmp_path), "cube.obj", 0.05)
    for enabled, generator in (
        (small, original.generate_small_cuboids),
        (big, original.generate_big_cuboids),
        (sticks, original.generate_sticks),
    ):
        if enabled:
            generator(str(tmp_path), "cube.obj", 0.05)
    names = sorted(path.name for path in tmp_path.glob("*.urdf"))
    if not randomize:
        names = names[:1]
    expected = [(name, tuple(float(value) / 100 for value in Path(name).stem.split("_")[2:])) for name in names]
    np.random.default_rng(42).shuffle(expected)
    port = load_module("dexpbt_port_catalog", TASK / "catalog.py")
    assert port.original_cube_catalog(randomize=randomize, small=small, big=big, sticks=sticks) == expected


def test_nonrandom_catalog_is_original_default_cube():
    port = load_module("dexpbt_port_catalog", TASK / "catalog.py")
    assert port.original_cube_catalog(randomize=False) == [("000_cube_100_100_100.urdf", (1.0, 1.0, 1.0))]
