"""CPU-only wrapper regressions: real Hydra and exec, SDK/Torch replaced by tiny stubs.

The temporary train entrypoint models the original Hydra/PBT boundary, not training.
Native initial-check/restart functions come from the immutable, manifest-pinned source
fixtures (665c32170d84b4be66722eea405a1e08b6e7f761), never an external checkout.
"""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dexpbt_original/train_native.py"
FIXTURE = ROOT / "tests/fixtures/dexpbt_original"

TRAIN_ENTRY = """
import ast
import json
import os
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch

# Only these definitions execute; native Torch/NumPy/RL-Games imports never run.
fixture = Path(os.environ['WRAPPER_TEST_FIXTURE']) / 'isaacgymenvs/pbt'
selected = {'_flatten_params', '_filter_params', 'PbtParams',
            '_restart_process_with_new_params', 'initial_pbt_check'}
tree = ast.parse((fixture / 'pbt.py').read_text())
definitions = [node for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected]
assert len(definitions) == len(selected)
RLAlgo = Any

def flatten_dict(values, prefix='', separator='.'):
    result = {}
    for key, value in values.items():
        if isinstance(value, dict):
            result.update(flatten_dict(value, prefix + key + separator, separator))
        else:
            result[prefix + key] = value
    return result

omegaconf_to_dict = lambda cfg: OmegaConf.to_container(cfg, resolve=True)
mutate = runpy.run_path(str(fixture / 'mutation.py'))['mutate']
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(fixture / 'pbt.py'), 'exec'))

@hydra.main(config_name='config', config_path='./cfg')
def launch_rlg_hydra(cfg: DictConfig):
    cycle = int(os.environ.get('WRAPPER_TEST_CYCLE', '0'))
    assert cycle <= 2, 'Unexpected restart loop'
    print('WRAPPER_PROBE ' + json.dumps({
        'cycle': cycle, 'pid': os.getpid(), 'argv': sys.argv,
        'executable': sys.executable, 'python3': shutil.which('python3'),
        'path_first': os.environ['PATH'].split(os.pathsep)[0],
        'source': os.environ['DEXPBT_NATIVE_SOURCE_ROOT'],
        'sdk': os.environ['DEXPBT_NATIVE_SDK_ROOT'],
        'fusion_env': os.environ.get('DEXPBT_NATIVE_DISABLE_GPU_FUSION'),
        'fusion': torch._C._jit_can_fuse_on_gpu(),
        'cfg': OmegaConf.to_container(cfg, resolve=True),
        'sources': OmegaConf.to_container(HydraConfig.get().runtime.config_sources),
        'cwd': os.getcwd(),
    }), flush=True)
    if not cfg.pbt.enabled:
        return
    os.environ['WRAPPER_TEST_CYCLE'] = str(cycle + 1)
    initial_pbt_check(cfg)  # Real initial mutation + exec on cycle 0; no-op thereafter.
    assert cycle > 0, 'Initial PBT check did not exec'
    if cycle == 1:
        _restart_process_with_new_params(
            cfg.pbt.policy_idx, {'learning_rate': 0.000123},
            str(Path.cwd() / 'donor.pth'), 'wrapper_test_member_' + str(cfg.pbt.policy_idx), None, False,
        )
        raise AssertionError('Exploit restart did not exec')
"""


@pytest.fixture
def native_source(tmp_path):
    """Create only temporary source/SDK stand-ins; preserve all reference fixtures."""
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    for name in ("pbt/pbt.py", "pbt/mutation.py", "cfg/pbt/pbt_default.yaml"):
        relative = "isaacgymenvs/" + name
        assert hashlib.sha256((FIXTURE / relative).read_bytes()).hexdigest() == manifest["files"][relative]

    source = tmp_path / "source"
    cfg = source / "isaacgymenvs/cfg"
    (cfg / "pbt/mutation").mkdir(parents=True)
    (source / "isaacgymenvs/train.py").write_text(textwrap.dedent(TRAIN_ENTRY))
    (cfg / "pbt/pbt_default.yaml").write_bytes((FIXTURE / "isaacgymenvs/cfg/pbt/pbt_default.yaml").read_bytes())
    (cfg / "pbt/mutation/default_mutation.yaml").write_text("learning_rate: mutate_float\n")
    (cfg / "config.yaml").write_text(textwrap.dedent("""
        defaults:
          - pbt: pbt_default
          - _self_
        task:
          name: cpu_wrapper_probe
          env:
            numEnvs: 4
        seed: 42
        learning_rate: 0.001
        checkpoint: ''
        wandb_activate: false
        hydra:
          output_subdir: null
          job:
            chdir: true
    """))
    sdk = tmp_path / "sdk"
    (sdk / "python").mkdir(parents=True)
    (sdk / "python/isaacgym.py").write_text("gymapi = object()\n")
    (sdk / "python/torch.py").write_text(textwrap.dedent("""
        import sys
        assert 'isaacgym' in sys.modules, 'SDK must import before Torch'
        __version__ = 'cpu-wrapper-stub'
        class Fusion:
            enabled = True
            def _jit_override_can_fuse_on_gpu(self, enabled):
                self.enabled = enabled
            def _jit_can_fuse_on_gpu(self):
                return self.enabled
        _C = Fusion()
    """))
    # Deliberately wrong PATH: passing an absolute Python executable alone is insufficient.
    wrong_bin = tmp_path / "wrong-bin"
    wrong_bin.mkdir()
    wrong_python = wrong_bin / "python3"
    wrong_python.write_text("#!/bin/sh\nexit 91\n")
    wrong_python.chmod(0o755)
    return source, sdk, wrong_bin


def _run_members(tmp_path, native_source, members, pbt):
    source, sdk, wrong_bin = native_source
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("DEXPBT_NATIVE_", "WRAPPER_TEST_"))
        and key not in ("HYDRA_MAIN_MODULE", "FB_PAR_MAIN_MODULE", "FB_XAR_MAIN_MODULE", "PYTHONHOME")
    }
    env.update(
        PATH=str(wrong_bin),
        PYTHONPATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        CUDA_VISIBLE_DEVICES="",
        WRAPPER_TEST_FIXTURE=str(FIXTURE),
    )
    processes = []
    outputs = []
    deadline = time.monotonic() + 30
    try:
        for member in members:
            command = [
                sys.executable,
                "-B",
                str(WRAPPER),
                "--source-root",
                str(source),
                "--sdk-root",
                str(sdk),
                "--disable-gpu-fusion",
                f"pbt.enabled={pbt}",
                "pbt.num_policies=4",
                f"pbt.policy_idx={member}",
                "pbt.mutation_rate=1.0",
                f"seed={42 + member}",
                f"hydra.run.dir={tmp_path / f'run-{member}'}",
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=tmp_path,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        for process in processes:
            output, _ = process.communicate(timeout=max(0.01, deadline - time.monotonic()))
            assert process.returncode == 0, output
            probes = [
                json.loads(line.removeprefix("WRAPPER_PROBE "))
                for line in output.splitlines()
                if line.startswith("WRAPPER_PROBE ")
            ]
            runtimes = [
                json.loads(line.removeprefix("NATIVE_RUNTIME "))
                for line in output.splitlines()
                if line.startswith("NATIVE_RUNTIME ")
            ]
            assert len(probes) == len(runtimes) == (3 if pbt else 1), output
            assert all(runtime["torch"] == "cpu-wrapper-stub" for runtime in runtimes)
            assert all(Path(runtime["torch_path"]) == sdk / "python/torch.py" for runtime in runtimes)
            outputs.append(probes)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
    return outputs


def test_default_hydra_config_resolves_beside_native_source(tmp_path, native_source):
    source, _, _ = native_source
    [[probe]] = _run_members(tmp_path, native_source, [0], pbt=False)
    assert probe["cfg"]["task"] == {"name": "cpu_wrapper_probe", "env": {"numEnvs": 4}}
    main = next(entry for entry in probe["sources"] if entry["provider"] == "main")
    assert main["schema"] == "file"
    assert Path(main["path"]) == source / "isaacgymenvs/cfg"


def test_four_members_preserve_runtime_and_config_through_two_native_execs(tmp_path, native_source):
    source, sdk, _ = native_source
    outputs = _run_members(tmp_path, native_source, range(4), pbt=True)
    assert len({probes[0]["pid"] for probes in outputs}) == 4
    for member, probes in enumerate(outputs):
        assert [probe["cycle"] for probe in probes] == [0, 1, 2]
        assert len({probe["pid"] for probe in probes}) == 1
        for probe in probes:
            assert probe["argv"][0] == str(WRAPPER)
            assert "--source-root" not in probe["argv"] and "--sdk-root" not in probe["argv"]
            assert probe["source"] == str(source) and probe["sdk"] == str(sdk)
            assert probe["fusion_env"] == "1" and probe["fusion"] is False
            assert Path(probe["executable"]).samefile(sys.executable)
            assert Path(probe["python3"]).samefile(sys.executable)
            assert Path(probe["path_first"]) == Path(probe["executable"]).parent
            assert probe["cfg"]["pbt"]["policy_idx"] == member
            assert probe["cfg"]["pbt"]["num_policies"] == 4
            assert probe["cfg"]["seed"] == 42 + member
            assert probe["cwd"] == str(tmp_path / f"run-{member}")
        initial, restarted, exploited = [probe["cfg"] for probe in probes]
        assert restarted["learning_rate"] != initial["learning_rate"]
        assert restarted == {**initial, "pbt_restart": True, "learning_rate": restarted["learning_rate"]}
        assert exploited == {
            **restarted,
            "learning_rate": 0.000123,
            "checkpoint": str(tmp_path / f"run-{member}/donor.pth"),
            "full_experiment_name": f"wrapper_test_member_{member}",
        }
