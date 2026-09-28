"""Launch untouched IsaacGymEnvs with a narrowly scoped Blackwell TorchScript compatibility switch.

Run with the isolated Python3.8/Torch PTX environment, never env_isaaclab.
The source task, rewards, physics and PPO/PBT remain in the pinned original checkout.
"""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=os.environ.get("DEXPBT_NATIVE_SOURCE_ROOT"))
    parser.add_argument("--sdk-root", type=Path, default=os.environ.get("DEXPBT_NATIVE_SDK_ROOT"))
    parser.add_argument(
        "--disable-gpu-fusion", action="store_true", help="Keep TorchScript, disable its GPU kernel fusion"
    )
    args, hydra_args = parser.parse_known_args()
    if args.source_root is None or args.sdk_root is None:
        parser.error("Provide --source-root and --sdk-root for the initial launch")
    source = args.source_root.resolve(strict=True)
    sdk = args.sdk_root.resolve(strict=True)
    os.environ["DEXPBT_NATIVE_SOURCE_ROOT"] = str(source)
    os.environ["DEXPBT_NATIVE_SDK_ROOT"] = str(sdk)
    # Upstream re-execs with argv[0]='python3'; resolve that name back to this environment.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    if args.disable_gpu_fusion:
        os.environ["DEXPBT_NATIVE_DISABLE_GPU_FUSION"] = "1"
    sys.path[:0] = [str(sdk / "python"), str(source)]
    # isort: off
    from isaacgym import gymapi  # noqa: F401
    import torch

    # isort: on

    if os.environ.get("DEXPBT_NATIVE_DISABLE_GPU_FUSION") == "1":
        torch._C._jit_override_can_fuse_on_gpu(False)
    print(
        "NATIVE_RUNTIME "
        + json.dumps(
            {
                "torch": torch.__version__,
                "torch_path": torch.__file__,
                "source_root": str(source),
                "sdk_root": str(sdk),
                "torchscript_enabled": True,
                "gpu_fusion_enabled": torch._C._jit_can_fuse_on_gpu(),
                "pid": os.getpid(),
            }
        ),
        flush=True,
    )
    # The original PBT re-exec uses argv[0]; retain this wrapper across every mutation.
    if not any(arg == "--config-path" or arg.startswith("--config-path=") for arg in hydra_args):
        hydra_args = ["--config-path", str(source / "isaacgymenvs/cfg"), *hydra_args]
    sys.argv = [str(Path(__file__).resolve()), *hydra_args]
    namespace = runpy.run_path(str(source / "isaacgymenvs/train.py"), run_name="dexpbt_native_entry")
    namespace["launch_rlg_hydra"]()


if __name__ == "__main__":
    main()
