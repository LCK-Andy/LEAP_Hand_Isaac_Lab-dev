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
parser.add_argument(
    "--arms",
    nargs="+",
    choices=[
        "baseline",
        "control",
        "selective",
        "copy-only",
        "gentler",
        "diversity",
        "maturity",
        "evidence",
        "cadence",
    ],
    default=["baseline", "control"],
)
parser.add_argument(
    "--gate-se",
    type=float,
    choices=[1.0, 2.0],
    default=2.0,
    help="Copy-only comparator gate; compare to baseline (1) or selective (2)",
)
parser.add_argument("--seed-base", type=int, default=42, help="Change per independent population replicate")
parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
args = parser.parse_args()
if args.envs <= 0 or args.epochs <= 0 or args.seed_base < 0 or len(set(args.arms)) != len(args.arms):
    parser.error("positive envs/epochs, nonnegative seed-base, and unique arms required")
if len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
    parser.error("GPU ids must be unique and nonnegative")
if args.launch and len(args.arms) > 2:
    parser.error("Launch at most two paired arms together; generate larger plans without --launch")
root = args.root.resolve()
root.mkdir(parents=True, exist_ok=True)
plans = []
for arm in (["probe"] if args.probe else args.arms):
    for member in range(1 if args.probe else 4):
        name = f"allegro-{arm}-{member}"
        gpu = args.gpus[(member // 2) % len(args.gpus)]
        rollout_frames = args.envs * 16
        steps = args.epochs * rollout_frames
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
            str(args.seed_base + member),
            "--headless",
            "pbt.enabled=true",
            f"pbt.selection_enabled={'false' if arm in ('control', 'probe') else 'true'}",
            "pbt.objective=allegro_goals",
            f"pbt.policy_idx={member}",
            "pbt.num_policies=4",
            f"pbt.directory={directory}",
            f"pbt.interval_steps={rollout_frames * (64 if arm == 'cadence' else 32)}",
            f"pbt.warmup_steps={rollout_frames * 192}",
            f"pbt.adaptation_steps={rollout_frames * 96}",
            "pbt.max_frame_lag=2097152",
            "pbt.max_score_age_seconds=1800",
            "pbt.min_completed_episodes=512",
            "pbt.threshold_abs=0.001",
            f"pbt.threshold_std={args.gate_se if arm == 'copy-only' else 2.0 if arm == 'selective' else 1.0}",
            "pbt.change_range=[1.1,1.5]",
            f"pbt.mutation_rate={0.33 if arm == 'gentler' else 1.0}",
            f"pbt.copy_only={'true' if arm == 'copy-only' else 'false'}",
            f"pbt.recipient_role={'weakest' if arm == 'diversity' else 'any'}",
            f"pbt.donor_maturity_rollouts={64 if arm == 'maturity' else 0}",
            f"pbt.evidence_windows={2 if arm == 'evidence' else 1}",
            "pbt.initial_diversity=true",
            'pbt.mutation={"agent.params.config.learning_rate":"mutate_float",'
            '"agent.params.config.grad_norm":"mutate_float",'
            '"agent.params.config.kl_threshold":"mutate_float"}',
            'pbt.mutation_bounds={"agent.params.config.learning_rate":[0.00001,0.001],'
            '"agent.params.config.grad_norm":[0.5,5.0],'
            '"agent.params.config.kl_threshold":[0.005,0.03]}',
            'params.env.obs_groups={"obs":["policy"],"states":["policy"]}',
            f"params.config.max_frames={steps}",
            "params.config.horizon_length=16",
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
                seed=args.seed_base + member,
                gpu=gpu,
                steps=steps,
                epochs=args.epochs,
                final_checkpoint=str(directory / f"logs/rl_games/allegro_hand/pbt_workspace/{member:03d}/final.pth"),
                metric="native_completed_goals_per_episode",
                pilot_only=True,
                script=str(script),
                log=str(log),
                argv=command,
            )
        )
# Preflight all session identities before launching anything.
for plan in plans:
    if (
        args.launch
        and subprocess.run(["tmux", "has-session", "-t", plan["session"]], capture_output=True).returncode == 0
    ):
        raise SystemExit(f"Session already exists: {plan['session']}")
(root / ("probe.json" if args.probe else "experiment.json")).write_text(json.dumps(plans, indent=2) + "\n")
if args.launch:
    for plan in plans:
        subprocess.run(["tmux", "new-session", "-d", "-s", plan["session"], "bash", plan["script"]], check=True)
print(json.dumps(plans, indent=2))
