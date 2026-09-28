"""Generate commands for the unmodified original Isaac Gym release, without launching it.

Unlike the Isaac Lab port, this route retains the original simulator. It requires
an independently installed, compatible Isaac Gym SDK/Python/GPU environment.
No installation, package changes, GPU allocation or network logging is performed.
"""

import argparse
import shlex
import subprocess
from pathlib import Path

COMMIT = "665c32170d84b4be66722eea405a1e08b6e7f761"


def commands(source_root, workspace, python, devices, population=8, frames=5_000_000_000, seed=42):
    source_root = Path(source_root).expanduser().resolve(strict=True)
    workspace = Path(workspace).expanduser().resolve()
    python = Path(python).expanduser().resolve(strict=True)
    if population < 4 or len(devices) != population or len(set(devices)) != population:
        raise ValueError("Require at least four members and one explicit distinct GPU ordinal per member")
    if any(not value.isdecimal() for value in devices) or frames <= 0 or seed < 0:
        raise ValueError("Require nonnegative GPU ordinals/seeds and a positive frame budget")
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError("Use an empty workspace; generated commands are fresh starts, not resume commands")
    head = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise ValueError(f"Original source must be pinned to {COMMIT}, found {head}")
    dirty = subprocess.check_output(["git", "-C", str(source_root), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise ValueError("Original source checkout must be clean for the native reproduction route")
    result = []
    for member, device in enumerate(devices):
        argv = [
            str(python),
            "-m",
            "isaacgymenvs.train",
            f"seed={seed + member}",
            "headless=True",
            "wandb_activate=False",
            "task=AllegroKukaLSTM",
            "task/env=reorientation",
            "num_envs=8192",
            "sim_device=cuda:0",
            "rl_device=cuda:0",
            "graphics_device_id=0",
            f"train.params.config.max_frames={frames}",
            "pbt=pbt_default",
            f"pbt.workspace={workspace}",
            "pbt.interval_steps=20000000",
            "pbt.start_after=100000000",
            "pbt.initial_delay=200000000",
            "pbt.replace_fraction_worst=0.3",
            "pbt/mutation=allegro_kuka_mutation",
            f"pbt.num_policies={population}",
            f"pbt.policy_idx={member}",
            f"experiment=dexpbt_original_member_{member:03d}",
        ]
        result.append(f"(cd {shlex.quote(str(source_root))} && CUDA_VISIBLE_DEVICES={device} {shlex.join(argv)})")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True, help="Compatible original-SDK Python, not env_isaaclab")
    parser.add_argument("--devices", required=True, help="Comma-separated distinct GPU ordinals")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=5_000_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = commands(
            args.source_root,
            args.workspace,
            args.python,
            args.devices.split(","),
            args.population,
            args.max_frames,
            args.seed,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print("# DRY RUN ONLY. Each command is a separate member; run concurrently in separate sessions.")
    print("# Requires compatible original Isaac Gym/PyTorch GPU stack; not validated on Blackwell.")
    print("# Source recipe, explicit 5B/member budget, deterministic seed identities, external logging disabled.")
    for command in result:
        print(command)


if __name__ == "__main__":
    main()
