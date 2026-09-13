"""
Benchmark the physics cost of the object collision proxies (spheres vs hull).

Steps Isaac-Repose-Objects-LEAP-Direct-v0 with seeded random actions and
reports steady-state env.step() throughput for one collision-proxy
configuration per process (identical action stream across configs):

  ~/env_isaaclab/bin/python scripts/bench_collision_proxy.py --proxy spheres14
  ~/env_isaaclab/bin/python scripts/bench_collision_proxy.py --proxy spheres50
  ~/env_isaaclab/bin/python scripts/bench_collision_proxy.py --proxy hull

Count PhysX patch-buffer overflow warnings from the process log afterwards:
  grep -ci "patch buffer overflow" <log>
"""

import argparse
import subprocess
import time
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).parent.parent

parser = argparse.ArgumentParser(description="Collision-proxy stepping benchmark.")
parser.add_argument("--task", default="Isaac-Repose-Objects-LEAP-Direct-v0")
parser.add_argument(
    "--proxy",
    choices=["spheres14", "spheres50", "hull"],
    required=True,
    help="collision configuration to benchmark",
)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=300, help="timed control steps")
parser.add_argument("--warmup", type=int, default=30, help="untimed warmup steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import LEAP_Isaaclab.tasks  # noqa: E402,F401  (registers the task)


def main():
    tag = f"bench:{args_cli.proxy}"
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.proxy == "hull":
        env_cfg.collision_proxy = "hull"
    elif args_cli.proxy == "spheres50":
        env_cfg.collision_proxy = "spheres"
        env_cfg.sphere_proxy_dir = str(
            REPO / "source/LEAP_Isaaclab/LEAP_Isaaclab/assets/ycb/spheres50"
        )
    # spheres14: cfg defaults

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped

    # guard: confirm the proxy actually in use (sphere count from one USD)
    if args_cli.proxy != "hull":
        from pxr import Usd, UsdGeom

        sphere_dir = env_cfg.sphere_proxy_dir or str(
            REPO / "source/LEAP_Isaaclab/LEAP_Isaaclab/assets/ycb/spheres"
        )
        stage = Usd.Stage.Open(f"{sphere_dir}/apple_spheres.usd")
        n = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Sphere))
        del stage
        expected = 14 if args_cli.proxy == "spheres14" else 50
        assert n == expected, f"apple proxy has {n} spheres, expected {expected}"
        print(f"[{tag}] proxy USD check OK: {n} spheres/object")

    torch.manual_seed(42)
    env.reset()
    with torch.inference_mode():
        for _ in range(args_cli.warmup):
            actions = 2 * torch.rand(env.action_space.shape, device=unwrapped.device) - 1
            env.step(actions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args_cli.steps):
            actions = 2 * torch.rand(env.action_space.shape, device=unwrapped.device) - 1
            _, reward, _, _, _ = env.step(actions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    env_steps = args_cli.steps * args_cli.num_envs
    print(f"[{tag}] envs={args_cli.num_envs} steps={args_cli.steps} wall={dt:.2f}s")
    print(f"[{tag}] {args_cli.steps / dt:.1f} control steps/s | {env_steps / dt:,.0f} env-steps/s")
    print(f"[{tag}] {dt / args_cli.steps * 1000:.2f} ms per control step")
    print(f"[{tag}] reward: mean {reward.mean().item():+.2f} min {reward.min().item():+.2f} (finite: {bool(torch.isfinite(reward).all())})")
    vram = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"[{tag}] VRAM used: {vram} MiB")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
