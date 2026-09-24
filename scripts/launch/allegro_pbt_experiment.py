"""Generate isolated, seed-paired official Allegro PBT/no-selection launches.

Run on the remote host with --root. Does not launch unless --launch is supplied.
The code snapshot and isolated rl-games 1.6.5 path must already exist.
"""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--python", default="/root/autodl-tmp/env_isaaclab/bin/python")
parser.add_argument("--deps", default="/root/dexpbt_scaled_20260924/python_deps")
parser.add_argument("--envs", type=int, default=1024)
parser.add_argument("--epochs", type=int, default=2048)
parser.add_argument("--probe", action="store_true")
parser.add_argument("--launch", action="store_true")
args = parser.parse_args()
root = args.root.resolve()
root.mkdir(parents=True, exist_ok=True)
plans = []
for arm in (["probe"] if args.probe else ["pbt", "control"]):
    for member in range(1 if args.probe else 4):
        name = f"allegro-{arm}-{member}"
        gpu = member // 2
        directory = root / arm
        log = directory / f"member_{member}.log"
        if log.exists():
            raise SystemExit(f"Refuse overwriting existing launch {log}")
        directory.mkdir(exist_ok=True)
        command = [
            args.python,
            "-u",
            str(root / "code/scripts/rl_games/train_dexpbt_member.py"),
            "--task",
            "Isaac-Repose-Cube-Allegro-Direct-v0",
            "--num_envs",
            str(args.envs),
            "--max_iterations",
            str(args.epochs),
            "--seed",
            str(42 + member),
            "--headless",
            "pbt.enabled=true",
            f"pbt.selection_enabled={'true' if arm == 'pbt' else 'false'}",
            "pbt.objective=allegro_goals",
            f"pbt.policy_idx={member}",
            "pbt.num_policies=4",
            f"pbt.directory={directory}",
            "pbt.interval_steps=524288",
            "pbt.warmup_steps=3145728",
            "pbt.adaptation_steps=1572864",
            "pbt.max_frame_lag=2097152",
            "pbt.max_score_age_seconds=1800",
            "pbt.min_completed_episodes=512",
            "pbt.threshold_abs=0.001",
            "pbt.threshold_std=1.0",
            "pbt.change_range=[1.1,1.5]",
            "pbt.mutation_rate=1.0",
            "pbt.initial_diversity=true",
            'pbt.mutation={"agent.params.config.learning_rate":"mutate_float",'
            '"agent.params.config.grad_norm":"mutate_float",'
            '"agent.params.config.kl_threshold":"mutate_float"}',
            'pbt.mutation_bounds={"agent.params.config.learning_rate":[0.00001,0.001],'
            '"agent.params.config.grad_norm":[0.5,5.0],'
            '"agent.params.config.kl_threshold":[0.005,0.03]}',
            'params.env.obs_groups={"obs":["policy"],"states":["policy"]}',
            "params.config.minibatch_size=8192",
            "params.config.save_frequency=512",
            "params.config.save_best_after=0",
            f"params.config.full_experiment_name=member_{member}",
        ]
        script = root / f"{arm}_{member}.sh"
        script.write_text(
            "#!/bin/bash\nset -uo pipefail\nulimit -c 0\n"
            f"export CUDA_VISIBLE_DEVICES={gpu}\n"
            f"export PYTHONPATH={shlex.quote(args.deps + ':' + str(root / 'code/source/LEAP_Isaaclab'))}\n"
            "export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1\n"
            f"cd {shlex.quote(str(root / 'code'))}\n"
            f"{shlex.join(command)} >> {shlex.quote(str(log))} 2>&1\n"
            f"rc=$?\nprintf '%s\\n' \"$rc\" > {log.with_suffix('.exit')}\nexit \"$rc\"\n"
        )
        plans.append(
            dict(
                session=name,
                arm=arm,
                member=member,
                seed=42 + member,
                gpu=gpu,
                steps=args.epochs * args.envs * 16,
                script=str(script),
                log=str(log),
                argv=command,
            )
        )
# Preflight all session identities before launching anything.
for plan in plans:
    if subprocess.run(["tmux", "has-session", "-t", plan["session"]], capture_output=True).returncode == 0:
        raise SystemExit(f"Session already exists: {plan['session']}")
(root / ("probe.json" if args.probe else "experiment.json")).write_text(json.dumps(plans, indent=2) + "\n")
if args.launch:
    for plan in plans:
        subprocess.run(["tmux", "new-session", "-d", "-s", plan["session"], "bash", plan["script"]], check=True)
print(json.dumps(plans, indent=2))
