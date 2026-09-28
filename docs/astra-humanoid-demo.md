# Astra controls a simulated humanoid

This is a small **online, tool-mediated control demo**, not reinforcement-learning training. The Astra model in a ZCode session reads simulator state and rendered images, selects short normalized joint-effort commands, and receives fresh observations. The Python bridge contains no language-model client, pretrained policy, gait generator, inverse kinematics, or extra balance controller.

## What “direct” means here

```text
Astra session → JSON effort command → validation → native env.step(action)
             ← measured state + rendered image ← PhysX simulation
```

The model must actually issue successive commands after observing feedback. Running the server alone does not run Astra. Replaying recorded commands would be a replay, not a new model-controlled run. The controller label in the manifest is supplied metadata, not authentication of a model identity.

Physics advances **only when a command calls `env.step()`**. Wall-clock model latency is excluded from simulation time. This demonstrates interactive decision-making; it does not establish a model capable of balancing a robot at real-time servo rates.

### Environment compatibility

The requested task is `Isaac-Humanoid-Direct-v0`. Installed Isaac Lab 3.0.0rc1 has removed that registration and supplies `Isaac-Humanoid-Direct`. The bridge resolves that one known rename when necessary, prints both names, and records them in `manifest.json`. It does not add a global alias or claim historical MDP equivalence.

The installed successor has 21 actions, an 87-dimensional policy observation, and a 60 Hz control interval (120 Hz physics, decimation 2). This demo explicitly selects `physics=isaacsim_physx`; the installed task otherwise defaults to Newton.

Actions are normalized **feedforward efforts**, not joint angles or Newton-metres:

```text
commanded feedforward torque = action × native joint gear × action_scale
```

Joint order and gears are read from the live articulation rather than hardcoded. The stock robot also has nonzero implicit stiffness and damping. Those passive drives remain enabled, so the model's feedforward effort is **not the entire applied torque**. Gravity, contacts, floating base, reward, and fall termination remain native. `--neutral-reset` optionally disables joint reset noise to make a basic gesture easier to interpret; it does not fix the root or provide external support.

## Launch

From the repository root, in one terminal:

```bash
/home/andy/env_isaaclab/bin/python scripts/demos/astra_humanoid.py \
  --run-dir outputs/astra_humanoid/my-demo \
  --neutral-reset \
  --controller-label 'GPT-6 Astra via my ZCode session'
```

Add `--gui` to display the Kit viewport. The default is headless with RGB capture. Startup requires the local Isaac Sim environment and the native humanoid USD asset (download/cache). A new run directory is mandatory; existing directories are never overwritten.

In another terminal, or as tools issued by the model:

```bash
python scripts/demos/astra_humanoid_protocol.py \
  --run-dir outputs/astra_humanoid/my-demo status

# Example syntax only: select real joint names and efforts from observed state.
python scripts/demos/astra_humanoid_protocol.py \
  --run-dir outputs/astra_humanoid/my-demo \
  action --torques '{}' --steps 3 --label 'Measure a short zero-effort baseline'

python scripts/demos/astra_humanoid_protocol.py \
  --run-dir outputs/astra_humanoid/my-demo close
```

`status` includes joint names, positions, velocities, limits, body positions, root orientation/velocity, policy observations, and the current PNG path. The model can read that image and choose its next action. `reset` starts another episode. Unspecified joints receive zero feedforward effort; actions do not retain omitted values from the preceding command.

### Bounded execution

- Reject unknown joints, malformed commands, nonfinite efforts, and efforts outside `[-1, 1]`.
- Default limit: 30 control steps per command (0.5 simulated seconds); CLI maximum: 60.
- Default lifetime budget: 600 control steps (10 simulated seconds), across resets.
- Stop a chunk immediately on a terminal transition. Native `env.step` has already auto-reset; returned state is explicitly marked as post-reset, with a separate `state_episode` identifier. Preserve native terminal policy observations. A subsequent reset command acknowledges this reset without resampling before more actions are accepted. The terminal image is **not** preserved: the corresponding image shows the reset state and must not be presented as the fall frame.
- Close after 15 minutes with no accepted command. No network listener or model credentials are used.
- Client timeout does **not** cancel an already submitted request. Inspect its matching response before retrying.

## Artifacts

Each run records:

- `manifest.json`: requested/resolved environment, seed, physics, passive gains and effort scale.
- `state.json`: latest measured state and image path.
- `commands/*.json` and `responses/*.json`: matched request/response pairs.
- `commands.jsonl`: submitted commands with their pre-action observations and labels.
- `transitions.jsonl`: every executed action, reward, state, and terminal flags.
- `frames/*.png`: reset and per-step frames, in acquisition order.
- `summary.json`: shutdown reason and step/frame counts; `error.json` if startup/execution failed.

Frame sequences include reset images. Encode per-episode segments if making a video; do not present reset jumps as physical motion. Playback at 60 fps represents simulation time, not model wall-clock speed.

## Measured local demo — 2026-09-28

Recorded run: `outputs/astra_humanoid/demo_20260928_r2/`.
Video: [`../outputs/astra_humanoid/astra_humanoid_demo.mp4`](../outputs/astra_humanoid/astra_humanoid_demo.mp4).

The GPT-6 Astra model hosting this ZCode session chose **eight successive action chunks** after inspecting joint/root feedback, with RGB inspections during the episode. This was not an offline generated controller or prerecorded action sequence. One earlier 28-step exploratory run informed the directions of the efforts. No separate model API or credentials were needed.

- The right shoulder's axis-2 angle moved from neutral to approximately **−49°**, then reversed under subsequent model commands. This establishes actuator control and a visible arm gesture, not a task success.
- Balance failed after **73 steps / 1.217 simulated seconds**. Terminal root height was **0.778 m**, below the native 0.8 m fall threshold. No walking success is claimed.
- A subsequent zero-feedforward baseline, using the same neutral reset option and stock passive drives, fell after **58 steps / 0.967 seconds**. This single informal comparison does not establish a robust balance improvement.
- The corrected run executed 132 total steps, including reset regression and baseline, then closed successfully with exit code 0. The earlier exploratory run also closed cleanly.
- Action → reset → action passed in live PhysX. A 12-step command stopped after five steps on termination; another action was refused until reset acknowledgment. Acknowledgment preserved the native autoreset pose, and termination flags survived close.
- The video plays at **one-fifth simulation speed** for visibility. It includes the initial pose and all preterminal gesture/fall frames, excluding the terminal autoreset image. The video is not model wall-clock timing and does not hide a successful recovery afterward.

The bridge/client protocol has 23 simulator-free unit tests, passing under both Python 3.12 and 3.14. A final-source 90-step PhysX integration smoke (`outputs/astra_humanoid/final_smoke_20260928/checks.json`) also passed action/reset/action, terminal-observation retention, terminal action rejection, autoreset acknowledgment, stale-final-observation clearing, and exact lifetime-budget enforcement, then exited with code 0. One test-harness expectation incorrectly treated filling the remaining budget as exceeding it; the corrected check allowed step 90 and rejected step 91. Black and targeted flake8 checks passed.

Headless RGB rendering was exercised live; GUI mode and its native reset button were not visually validated. Use the JSON reset command rather than viewport controls for recorded experiments. All simulator processes launched for this demo were closed afterward.

## Related examples

The user-provided [Awesome Astra Embodied AI collection](https://github.com/zjwzcx/Awesome-Astra-Embodied-AI) includes materially different control architectures:

- [GPT-Policy](https://github.com/cheng-haha/GPT-Policy-Eval): online model tool actions with conventional trajectory/IK execution.
- [GPT-as-Policy direct mode](https://github.com/anonymous-report-421/GPT-as-Policy/blob/main/hybrid_rollout/robodojo/robodojo-gpt-only-rollout/context/eef_control.md): short bounded end-effector commands with local IK.
- [HumanCLAW](https://github.com/Human-CLAW/HumanCLAW): visual skill selection above pretrained motion generation.
- [DexLab apple-stem demo](https://github.com/huangkiki/Dexlab/tree/main/demos/apple-stem-grasp): model-assisted controller development, explicitly no model API at runtime.

This demo uses the existing ZCode session as the online model host rather than installing an additional API harness. It demonstrates a narrower interface than high-level humanoid navigation, and does not claim the locomotion task is solved.
