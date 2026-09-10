#!/usr/bin/env bash
# Train the official Isaac-Dexsuite-Kuka-Allegro-Lift task with FlashSAC (flash_rl)
# inside env_isaaclab.
#
# The stock task id is MANAGER-BASED with the state split across three obs
# groups (policy / proprio / perception); flash_rl only reads "policy", so this
# trains the DIRECT port registered in LEAP_Isaaclab.tasks.dexsuite_lift_direct
# (tasks/dexsuite_lift_direct/): the official Lift MDP (relative joint-position
# actions scale 0.1, official rewards with per-fingertip ContactSensor gates,
# ADR gravity/noise curriculum) as a DirectRLEnv with one flat 785-d policy
# observation (157 x 5 history).
#
# Usage:
#   scripts/flashsac_train_dexsuite_lift.sh
# Tunables (env vars), laptop profile defaults (12 GB RTX 5070 Ti + 30 GB RAM):
#   NUM_ENV_STEPS=1004857600 NUM_TRAIN_ENVS=512 BUFFER_LEN=2000000 BUFFER_DEVICE=cpu \
#   UPDATES_PER_STEP=1 SAVE_EVERY=10000 GROUP=dexsuite EXP=lift-r1
#   (SAVE_EVERY is in control iterations; default = one checkpoint per
#    ~10.5M env-steps; train.py also always writes a "final" checkpoint.)
# Behavior knob: ACT_SMOOTH=0.3 applies joint-target EMA smoothing like the
#   LEAP tasks (default 1.0 = off, matching the official dexsuite task).
# HOLD_TIME stays 0: the hold-to-confirm hook only exists for the direct
#   in-hand task and is inactive here.
# Extra hydra overrides pass through as script args, e.g.:
#   scripts/flashsac_train_dexsuite_lift.sh --overrides agent.compile_mode=default
#   (use that if training crashes with the intermittent torch cudagraph-trees
#    "live storage data ptrs are in the cudagraph pool" error)
# Replay-buffer placement (VRAM vs RAM): the flash_rl buffer supports BOTH --
#   BUFFER_DEVICE=cuda  buffer lives in VRAM (fastest sampling)
#   BUFFER_DEVICE=cpu   buffer lives in PINNED system RAM; each update copies
#                       only the 2048-row mini-batch to the GPU asynchronously
#                       (~13 MB over PCIe, a few ms) -- the 12 GB GPU then only
#                       pays for the sim (~4.6 GB at 512 envs) + networks, so a
#                       much larger buffer becomes affordable.
# RAM budget (30 GB laptop, pinned memory cannot swap; Isaac Sim's host process
# takes ~8-10 GB itself): each 1M transitions of the 785-d obs costs ~5.9 GB
# (obs + next_obs, float32). The 2M default pins ~12.6 GB -- the practical
# ceiling; if the process gets host-OOM-killed or the system stalls in the first
# minutes, drop to BUFFER_LEN=1500000. The LEAP tasks' 6M buffer needed 35 GB
# here -- impossible even in RAM alone.
# NOTE: these `${VAR:-default}` tunables HONOR values already exported in your
# shell (e.g. a stale BUFFER_LEN=6000000 from a LEAP launch) -- the banner below
# prints the effective profile so mismatches are visible.

set -euo pipefail

FLASHSAC_DIR="${FLASHSAC_DIR:-$HOME/Documents/projects/FlashSAC}"
PYTHON="${PYTHON:-$HOME/env_isaaclab/bin/python}"
TASK="${TASK:-Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-1004857600}"
NUM_TRAIN_ENVS="${NUM_TRAIN_ENVS:-1024}"
BUFFER_LEN="${BUFFER_LEN:-2000000}"
BUFFER_DEVICE="${BUFFER_DEVICE:-cpu}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-1}"
SAVE_EVERY="${SAVE_EVERY:-$((10485760 / NUM_TRAIN_ENVS))}"
# upstream derives logging/metrics cadence from the TOTAL step budget
# (logging = num_interaction_steps/100, metrics = /10), which at a ~1B-step
# budget means one actor/critic point every ~10M env steps (~30 min) and one
# avg_return point every ~100M (~5 h). Pin them to wall-clock-friendly cadences
# instead: LOG_EVERY ~1M env steps, METRICS_EVERY ~4M (both in CONTROL
# ITERATIONS, like SAVE_EVERY).
LOG_EVERY="${LOG_EVERY:-$((1048576 / NUM_TRAIN_ENVS))}"
METRICS_EVERY="${METRICS_EVERY:-$((4194304 / NUM_TRAIN_ENVS))}"
HOLD_TIME="${HOLD_TIME:-0}"
ACT_SMOOTH="${ACT_SMOOTH:-1.0}"
GROUP="${GROUP:-dexsuite}"
EXP="${EXP:-lift-r1}"
EXTRA_OVERRIDES=("$@")

echo "[flashsac-dexsuite-lift] task=$TASK envs=$NUM_TRAIN_ENVS buffer=$BUFFER_LEN@$BUFFER_DEVICE " \
     "steps=$NUM_ENV_STEPS save_every_iters=$SAVE_EVERY log_every_iters=$LOG_EVERY " \
     "metrics_every_iters=$METRICS_EVERY group=$GROUP exp=$EXP"

cd "$FLASHSAC_DIR"
# registers LEAP_Isaaclab.tasks (incl. the direct dexsuite Lift port) post-launch
export FLASH_RL_EXTRA_TASK_MODULES="${FLASH_RL_EXTRA_TASK_MODULES:-LEAP_Isaaclab.tasks}"
# in-hand-only hold hook: disabled (see header); EMA smoothing stays available
export FLASH_RL_HOLD_TIME="$HOLD_TIME"
export FLASH_RL_ACT_MOVING_AVERAGE="$ACT_SMOOTH"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export __NV_PRIME_RENDER_OFFLOAD=1

exec "$PYTHON" train.py \
    --config_name flashSAC_base \
    --overrides env=isaaclab \
    --overrides "env.env_name=$TASK" \
    --overrides "num_env_steps=$NUM_ENV_STEPS" \
    --overrides "num_train_envs=$NUM_TRAIN_ENVS" \
    --overrides num_eval_envs=null \
    --overrides num_record_envs=null \
    --overrides "num_eval_episodes=$NUM_TRAIN_ENVS" \
    --overrides num_record_episodes=0 \
    --overrides recording_per_interaction_step=null \
    --overrides "agent.buffer_max_length=$BUFFER_LEN" \
    --overrides "agent.buffer_device_type=$BUFFER_DEVICE" \
    --overrides "logging_per_interaction_step=$LOG_EVERY" \
    --overrides "metrics_per_interaction_step=$METRICS_EVERY" \
    --overrides "save_checkpoint_per_interaction_step=$SAVE_EVERY" \
    --overrides "updates_per_interaction_step=$UPDATES_PER_STEP" \
    --overrides gamma=0.99 \
    --overrides n_step=3 \
    --overrides "group_name=$GROUP" \
    --overrides "exp_name=$EXP" \
    --overrides logger_type=tensorboard \
    ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}
