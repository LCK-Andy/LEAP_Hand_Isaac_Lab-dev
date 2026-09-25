"""Launch a bounded native LEAP contact PPO/PBT population on two GPUs.

Requires an isolated root/code snapshot. Refuses existing logs or tmux sessions.
Use --probe for one short member, or --launch to execute the printed commands.
"""

import argparse
import json
import math
import shlex
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--python", default="/root/autodl-tmp/env_isaaclab/bin/python")
parser.add_argument("--deps", default="/root/dexpbt_scaled_20260924/python_deps")
parser.add_argument("--envs", type=int, default=2048)
parser.add_argument("--members", type=int, choices=(4, 8), default=4)
parser.add_argument("--steps", type=int, default=50_000_000)
parser.add_argument("--probe", action="store_true")
parser.add_argument("--launch", action="store_true")
args = parser.parse_args()
root = args.root.resolve()
if not (root / "code/scripts/rl_games/train_dexpbt_member.py").is_file():
    raise SystemExit("Deploy root/code before launching")
if args.envs <= 0 or args.steps <= 0 or (args.envs * 16) % 8192:
    raise SystemExit("Positive budget and envs*16 divisible by minibatch 8192 required")
arm = "probe" if args.probe else "population"
directory = root / arm
plans = []
for member in range(1 if args.probe else args.members):
    session = f"leap-contact-{arm}-{member}"
    log = directory / f"member_{member}.log"
    script = root / f"{arm}_{member}.sh"
    if log.exists() or script.exists():
        raise SystemExit(f"Refuse overwriting {log} or {script}")
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise SystemExit(f"Session exists: {session}")
    epochs = 24 if args.probe else math.ceil(args.steps / (args.envs * 16))
    command = [
        args.python,
        "-u",
        str(root / "code/scripts/rl_games/train_dexpbt_member.py"),
        "--task",
        "Isaac-Contact-Cube-LEAP-Direct-v0",
        "--num_envs",
        str(args.envs),
        "--max_iterations",
        str(epochs),
        "--seed",
        str(42 + member),
        "--headless",
        f"pbt.policy_idx={member}",
        f"pbt.num_policies={args.members}",
        f"pbt.directory={directory}",
        f"params.config.full_experiment_name=member_{member}",
    ]
    if args.probe:
        command += ["pbt.interval_steps=131072", "pbt.selection_enabled=false"]
    plans.append(
        dict(
            session=session,
            member=member,
            seed=42 + member,
            gpu=member // (args.members // 2),
            steps=epochs * args.envs * 16,
            envs=args.envs,
            script=str(script),
            log=str(log),
            argv=command,
        )
    )
# All preflights precede any file writing / process launch.
directory.mkdir(parents=True, exist_ok=True)
for plan in plans:
    Path(plan["script"]).write_text(
        "#!/bin/bash\nset -uo pipefail\nulimit -c 0\n"
        f"export CUDA_VISIBLE_DEVICES={plan['gpu']}\n"
        f"export PYTHONPATH={shlex.quote(args.deps + ':' + str(root / 'code/source/LEAP_Isaaclab'))}\n"
        "export OMP_NUM_THREADS=2 HYDRA_FULL_ERROR=1 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1\n"
        f"cd {shlex.quote(str(root / 'code'))}\n"
        f"{shlex.join(plan['argv'])} >> {shlex.quote(plan['log'])} 2>&1\n"
        f"rc=$?\nprintf '%s\\n' \"$rc\" > {shlex.quote(str(Path(plan['log']).with_suffix('.exit')))}\nexit \"$rc\"\n"
    )
(root / f"{arm}.json").write_text(json.dumps(plans, indent=2) + "\n")
if args.launch:
    for plan in plans:
        subprocess.run(["tmux", "new-session", "-d", "-s", plan["session"], "bash", plan["script"]], check=True)
print(json.dumps(plans, indent=2))
