# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Exercise DexPBT episode metrics, curriculum restoration and fixed evaluation."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import json
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import LEAP_Isaaclab.tasks  # noqa: F401


def main():
    task = "Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.episode_length_s = 0.1
    env = gym.make(task, cfg=cfg)
    try:
        env.reset()
        raw = env.unwrapped
        assert raw.reward_manager.get_term_cfg("success").params["pos_std"] == 0.2
        state = raw.get_env_state()
        json.dumps(state, allow_nan=False)
        for fraction in (1.0, 0.0):
            raw.configure_fixed_evaluation(fraction)
            env.reset()
            condition = raw.get_evaluation_condition()
            assert abs(condition["adr_fraction"] - fraction) < 1e-6, condition
            assert condition["curriculum_tolerance"] == 0.1
            assert raw.evaluation_protocol == "heldout_fixed_definition_v1"
            completed = 0
            for _ in range(12):
                _, reward, _, _, info = env.step(torch.zeros((args.num_envs, 23), device=raw.device))
                assert torch.isfinite(reward).all()
                report = info["dexpbt"]
                completed += report["completed_episodes"]
                assert 0 <= report["success_count"] <= report["completed_episodes"]
                json.dumps(report, allow_nan=False)
            assert completed >= args.num_envs, completed
        raw.set_env_state(state)
        assert raw.evaluation_protocol == "training_conditions_fixed_definition_v1"
        assert raw.reward_manager.get_term_cfg("success").params["pos_std"] == 0.2
        assert raw.get_env_state() == state
        print("DEXPBT_SMOKE_PASS: complete episode metrics, fixed ADR 1/0, curriculum state restoration")
    finally:
        env.close()


if __name__ == "__main__":
    import sys
    import traceback

    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close(exit_code=exit_code)
    sys.exit(exit_code)
