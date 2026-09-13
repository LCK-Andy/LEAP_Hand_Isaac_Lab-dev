"""Headless smoke test: the Direct-RL Dexsuite KUKA+Allegro Lift port through the REAL flash_rl wrapper.

Exercises the customized direct task end-to-end (flash_rl.envs.isaaclab, not a
direct gym make) to verify:
  * the task registers via FLASH_RL_EXTRA_TASK_MODULES (LEAP_Isaaclab.tasks)
  * action bounds 0.1 and the 23-d action space (KUKA 7 + Allegro 16)
  * the flat policy observation: 160/step x 5 history = 800-d
  * reset/step paths, finite rewards, terminations, forced reset (ADR update)

Usage:
  python scripts/flashsac_dexsuite_lift_smoke.py
"""

import os
import sys

import torch

os.environ.setdefault("FLASH_RL_EXTRA_TASK_MODULES", "LEAP_Isaaclab.tasks")

from flash_rl.envs.isaaclab import ACTION_BOUNDS, make_isaaclab_env  # noqa: E402

TASK = "Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0"
NUM_ENVS = 8
OBS_DIM = 5 * (4 + 3 + 7 + 23 + 23 + 23 + 5 * 13 + 4 * 3)  # 800
ACTION_DIM = 23


def main() -> None:
    assert ACTION_BOUNDS.get(TASK) == 0.1, f"action bounds not registered for {TASK}: {ACTION_BOUNDS.get(TASK)}"

    env = make_isaaclab_env(env_name=TASK, num_envs=NUM_ENVS, seed=0, headless=True)
    unwrapped = env.envs.unwrapped

    obs, info = env.reset()
    assert obs.shape == (NUM_ENVS, OBS_DIM), f"obs shape {obs.shape} != ({NUM_ENVS}, {OBS_DIM})"
    assert env.action_size == (ACTION_DIM,), f"action size {env.action_size} != {(ACTION_DIM,)}"
    assert info["actor_observation_size"] == (OBS_DIM,)
    print(f"[smoke] reset ok: obs {tuple(obs.shape)}, action_size {env.action_size}, bounds {env.action_bounds}")

    # a few random steps (uniform in [-1, 1] like the wrapper-clamped policy output)
    generator = torch.Generator().manual_seed(0)
    for step in range(50):
        actions = torch.rand(NUM_ENVS, ACTION_DIM, generator=generator) * 2.0 - 1.0
        obs, rew, term, trunc, info = env.step(actions)
        assert obs.shape == (NUM_ENVS, OBS_DIM), f"step {step}: obs {obs.shape}"
        assert rew.shape == (NUM_ENVS,) and term.shape == (NUM_ENVS,), f"step {step}: rew {rew.shape}"
        assert not torch.isnan(torch.as_tensor(rew)).any(), f"step {step}: NaN reward"
    print(f"[smoke] 50 random steps ok: rewards [{rew.min():.4f}, {rew.max():.4f}], difficulty "
          f"{unwrapped._difficulty.float().mean().item():.2f}")

    # force a reset to exercise the ADR update + reset writes
    env_ids = torch.arange(NUM_ENVS, device=unwrapped.device)
    unwrapped._reset_idx(env_ids)
    obs, rew, term, trunc, info = env.step(torch.zeros(NUM_ENVS, ACTION_DIM))
    assert obs.shape == (NUM_ENVS, OBS_DIM)
    print("[smoke] forced reset + ADR update ok")

    print("[smoke] PASSED")


if __name__ == "__main__":
    sys.exit(main())
