# Copyright (c) 2026, LCK-Andy.
# SPDX-License-Identifier: BSD-3-Clause
"""Print, but never execute, original DexPBT population launch commands.

Requires an explicit unique device per member. Example:
  python scripts/dexpbt_original/launch.py --devices 0,1,2,3,4,5,6,7 --workspace /data/original
The generator uses the paper experiment budget5B/member explicitly, overriding
released YAML10B. Redirect output to a reviewed shell script if desired.
"""

import argparse
import shlex
import sys
from pathlib import Path


def commands(
    workspace,
    devices,
    population=8,
    profile="released_source",
    python=sys.executable,
    num_envs=8192,
    max_frames=5_000_000_000,
    seed=42,
    allow_physics_mismatch=False,
):
    workspace = Path(workspace).expanduser().resolve()
    if population < 4 or len(devices) != population or len(set(devices)) != population:
        raise ValueError("Supply one unique GPU ordinal per population member (at least4); no implicit GPU sharing")
    if any(not device.isdecimal() for device in devices):
        raise ValueError("--devices must list nonnegative CUDA ordinals")
    if profile not in {"released_source", "paper"} or num_envs <= 0 or max_frames <= 0:
        raise ValueError("Invalid profile, environment count or frame budget")
    if (num_envs * 16) % 32768:
        raise ValueError("num_envs must fit the source32768 minibatch; use train.py explicit overrides for tiny checks")
    for member in range(population):
        directory = workspace / f"{member:03d}"
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Refusing fresh commands for populated member directory {directory}")
    train = Path(__file__).resolve().with_name("train.py")
    return [
        [
            python,
            str(train),
            "--headless",
            f"--device=cuda:{device}",
            f"--workspace={workspace}",
            f"--member={member}",
            f"--population={population}",
            f"--profile={profile}",
            f"--seed={seed + member}",
            f"--num_envs={num_envs}",
            f"--max-frames={max_frames}",
            *(["--allow-physics-mismatch"] if allow_physics_mismatch else []),
        ]
        for member, device in enumerate(devices)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--devices", required=True, help="Explicit comma-separated GPU ordinals, one per member")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--profile", choices=["released_source", "paper"], default="released_source")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--max-frames", type=int, default=5_000_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-physics-mismatch",
        action="store_true",
        help="Explicitly opt generated commands into the experimental physics-mismatched port",
    )
    args = parser.parse_args()
    try:
        result = commands(
            args.workspace,
            args.devices.split(","),
            args.population,
            args.profile,
            args.python,
            args.num_envs,
            args.max_frames,
            args.seed,
            args.allow_physics_mismatch,
        )
    except ValueError as error:
        parser.error(str(error))
    if not args.allow_physics_mismatch:
        print(
            "# BLOCKED: commands require explicit --allow-physics-mismatch; modern PhysX does not match original signed inertia."
        )
    print("# DRY RUN ONLY: run each command in its own process/session on the explicitly assigned GPU.")
    print("# 5B/member default is the paper budget; task + default PBT behavior remain released_source.")
    for command in result:
        print(shlex.join(command))


if __name__ == "__main__":
    main()
