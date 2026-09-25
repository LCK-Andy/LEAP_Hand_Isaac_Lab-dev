"""Catch removed native task imports without importing pxr before AppLauncher."""

import ast
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class EATaskImportTests(unittest.TestCase):
    def test_repose_spawn_namespace_is_segment_safe(self):
        path = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_repose/leap_direct_env_cfg.py"
        tree = ast.parse(path.read_text())
        paths = [
            node.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.keyword) and node.arg == "prim_path"
            and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            and node.value.value.startswith("/World/envs/")
        ]
        self.assertTrue(paths)
        self.assertTrue(all("env_[^/]+/" in path for path in paths), paths)

    def test_native_task_modules_exist(self):
        spec = importlib.util.find_spec("isaaclab_tasks")
        if spec is None:
            self.skipTest("requires the installed Isaac Lab task distribution")
        roots = [Path(path) for path in spec.submodule_search_locations]
        missing = []
        for directory in (ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab", ROOT / "scripts"):
            for path in directory.rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                    modules = []
                    if isinstance(node, ast.ImportFrom) and node.module:
                        modules = [node.module]
                    elif isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    for module in modules:
                        if not module.startswith("isaaclab_tasks."):
                            continue
                        relative = Path(*module.split(".")[1:])
                        if not any(
                            (root / relative).is_dir() or (root / relative).with_suffix(".py").is_file()
                            for root in roots
                        ):
                            missing.append(f"{path.relative_to(ROOT)}:{node.lineno}: {module}")
        self.assertEqual(missing, [], "Removed native task imports:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
