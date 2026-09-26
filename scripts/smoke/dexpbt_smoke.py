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
    # Exercise the native six-second episode and solver, not a shortened MDP.
    assert cfg.episode_length_s == 6.0
    assert cfg.scene.robot.spawn.articulation_props[0].solver_position_iteration_count == 32
    env = gym.make(task, cfg=cfg)
    try:
        env.reset()
        raw = env.unwrapped
        assert raw.reward_manager.get_term_cfg("success").params["pos_std"] == 0.2
        state = raw.get_env_state()
        json.dumps(state, allow_nan=False)
        from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt.pbt_cfg import PbtCfg
        from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.pbt.pbt_utils import EpisodeWindow

        window_cfg = PbtCfg(min_completed_episodes=1, max_score_age_seconds=10.0)
        window = EpisodeWindow(window_cfg)
        first_report = None
        for step in range(raw.max_episode_length + 1):
            _, reward, terminated, truncated, info = env.step(torch.zeros((args.num_envs, 23), device=raw.device))
            assert torch.isfinite(reward).all()
            report = info["dexpbt"]
            if report["completed_episodes"]:
                assert report["completed_episodes"] == int((terminated | truncated).sum())
                assert report["mixed_condition_episodes"] == 0
                if first_report is None:
                    first_report = json.loads(json.dumps(report))
                window.add(info, frame=(step + 1) * args.num_envs, now=100.0 + step * 0.001)
        assert first_report is not None
        frame = (raw.max_episode_length + 1) * args.num_envs
        ready = window.ready(window_cfg, frame=frame, now=101.0)
        assert ready and sum(group["completed_episodes"] for group in ready.values()) >= 1
        print("DEXPBT_TRAIN_EPISODES", sum(group["completed_episodes"] for group in ready.values()), flush=True)
        # Age genuine completed episode evidence; publication must retain only the
        # later batch, never refresh stale observations or relabel partial episodes.
        window.add({"dexpbt": first_report}, frame=frame + 1, now=109.0)
        ready = window.ready(window_cfg, frame=frame + 1, now=111.0)
        assert sum(group["completed_episodes"] for group in ready.values()) == first_report["completed_episodes"]
        assert all(group["created_at"] == 109.0 for group in ready.values())
        assert window.diagnostics["pruned_episodes"]["expired_evidence"] > 0
        enough = sum(group["completed_episodes"] for group in ready.values())
        strict_cfg = PbtCfg(min_completed_episodes=enough + 1, max_score_age_seconds=10.0)
        assert not window.ready(strict_cfg, frame=frame + 1, now=111.0)
        assert window.diagnostics["reason"] == "insufficient_fresh_episodes"
        for fraction in (1.0, 0.0):
            raw.configure_fixed_evaluation(fraction)
            env.reset()
            condition = raw.get_evaluation_condition()
            assert abs(condition["adr_fraction"] - fraction) < 1e-6, condition
            assert condition["curriculum_tolerance"] == 0.1
            assert raw.evaluation_protocol == "heldout_fixed_definition_v1"
            completed = 0
            for _ in range(raw.max_episode_length + 1):
                _, reward, _, _, info = env.step(torch.zeros((args.num_envs, 23), device=raw.device))
                assert torch.isfinite(reward).all()
                report = info["dexpbt"]
                completed += report["completed_episodes"]
                assert 0 <= report["success_count"] <= report["completed_episodes"]
                assert report["mixed_condition_episodes"] == 0
                json.dumps(report, allow_nan=False)
                # A training window must not accept heldout-condition reports.
                if report["completed_episodes"]:
                    heldout_window = EpisodeWindow(window_cfg)
                    heldout_window.add(info, frame=frame, now=101.0)
                    assert not heldout_window.ready(window_cfg, frame=frame, now=101.0)
            assert completed >= args.num_envs, completed
            print("DEXPBT_FIXED_EPISODES", fraction, completed, flush=True)
        raw.set_env_state(state)
        assert raw.evaluation_protocol == "training_conditions_fixed_definition_v1"
        assert raw.reward_manager.get_term_cfg("success").params["pos_std"] == 0.2
        assert raw.get_env_state() == state
        print("DEXPBT_SMOKE_PASS: native episodes/solver32, fresh completed evidence, fixed ADR 1/0, state restoration")
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
