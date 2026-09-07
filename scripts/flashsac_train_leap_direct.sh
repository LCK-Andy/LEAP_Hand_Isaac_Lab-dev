#!/usr/bin/env bash
# Train Isaac-Repose-Cube-LEAP-Direct-v0 with FlashSAC (flash_rl) inside env_isaaclab.
#
# FlashSAC lives in a patched clone at ~/Documents/projects/FlashSAC (see AGENTS.md
# "FlashSAC" for the local modifications): optional jax, custom-task registration
# via FLASH_RL_EXTRA_TASK_MODULES, and the LEAP task's action bounds.
#
# This profile is tuned for a 12 GB laptop GPU (RTX 5070 Ti): 512 envs, 1M CUDA
# replay buffer, UTD 1, n-step 3. The upstream Isaac Lab profile is 1024 envs /
# 10M buffer / UTD 2 -- scale up on a bigger GPU.
#
# Behavior knobs (applied inside the env by flash_rl, also active in play):
#   HOLD_TIME="0.5:1.5"  hold-to-confirm goals: the object must keep the target
#                        orientation this long before the goal counts/resamples.
#                        "lo:hi" randomizes per env and per goal; "1.0" is fixed;
#                        "0" disables (upstream immediate switching).
#   ACT_SMOOTH=0.3       joint-target EMA smoothing (act_moving_average); lower is
#                        smoother/slower; 1.0 disables (upstream unfiltered).
#
# Usage:
#   scripts/flashsac_train_leap_direct.sh
# Tunables (env vars):
#   NUM_ENV_STEPS=10485760 NUM_TRAIN_ENVS=512 BUFFER_LEN=1000000 \
#   UPDATES_PER_STEP=1 GROUP=leap_direct EXP=run1
# Extra hydra overrides are passed through as script args, e.g.:
#   scripts/flashsac_train_leap_direct.sh --overrides agent.use_compile=false

set -euo pipefail

FLASHSAC_DIR="${FLASHSAC_DIR:-$HOME/Documents/projects/FlashSAC}"
PYTHON="${PYTHON:-$HOME/env_isaaclab/bin/python}"
TASK="${TASK:-Isaac-Repose-Cube-LEAP-Direct-v0}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-400485760}"
NUM_TRAIN_ENVS="${NUM_TRAIN_ENVS:-4096}"
BUFFER_LEN="${BUFFER_LEN:-6000000}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-1}"
SAVE_EVERY="${SAVE_EVERY:-2048}"
HOLD_TIME="${HOLD_TIME:-0.2:1.5}"
ACT_SMOOTH="${ACT_SMOOTH:-0.2}"
GROUP="${GROUP:-leap_direct}"
EXP="${EXP:-run1}"
EXTRA_OVERRIDES=("$@")

cd "$FLASHSAC_DIR"
export FLASH_RL_EXTRA_TASK_MODULES="${FLASH_RL_EXTRA_TASK_MODULES:-LEAP_Isaaclab.tasks}"
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
    --overrides "save_checkpoint_per_interaction_step=$SAVE_EVERY" \
    --overrides "updates_per_interaction_step=$UPDATES_PER_STEP" \
    --overrides gamma=0.99 \
    --overrides n_step=3 \
    --overrides "group_name=$GROUP" \
    --overrides "exp_name=$EXP" \
    --overrides logger_type=tensorboard \
    ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}
