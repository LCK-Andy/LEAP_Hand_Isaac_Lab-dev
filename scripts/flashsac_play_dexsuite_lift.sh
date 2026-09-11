#!/usr/bin/env bash
# Play a FlashSAC-trained Dexsuite KUKA+Allegro Lift policy with the Isaac Sim GUI.
#
# Defaults: newest step*/ checkpoint of GROUP=dexsuite EXP=lift-r1 (the
# flashsac_train_dexsuite_lift.sh profile), played on the -Direct-Play-v0 env
# (ADR pinned at max difficulty: full gravity + full obs noise -- the real
# eval). Set PLAY_TASK=Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0 to watch the
# training-time env instead (difficulty starts at 0 and gravity is zero, so the
# object floats mid-air -- it looks trivial by design).
#
# Play only AFTER training has exited (a live training run + the GUI both want
# the GPU). The goal pose is not drawn in the GUI (the direct port has no goal
# marker) -- judge by the arm picking the cube off the table and lifting it;
# episode stats print in the console.
#
# Usage:
#   scripts/flashsac_play_dexsuite_lift.sh                          # newest ckpt, 8 envs, 50 episodes
#   scripts/flashsac_play_dexsuite_lift.sh -n 4 --num-episodes 200
#   scripts/flashsac_play_dexsuite_lift.sh --checkpoint ~/Documents/projects/FlashSAC/models/dexsuite/lift-r1/Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0/seed0-0910-101558/step81920
#   EXP=lift-r1 scripts/flashsac_play_dexsuite_lift.sh              # non-default experiment name

set -euo pipefail

FLASHSAC_DIR="${FLASHSAC_DIR:-$HOME/Documents/projects/FlashSAC}"
PYTHON="${PYTHON:-$HOME/env_isaaclab/bin/python}"
GROUP="${GROUP:-dexsuite}"
EXP="${EXP:-lift-r1}"
TRAIN_TASK="${TRAIN_TASK:-Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0}"  # checkpoint dirs live under the train id
PLAY_TASK="${PLAY_TASK:-Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-Play-v0}"  # env actually stepped
NUM_ENVS="${NUM_ENVS:-8}"
NUM_EPISODES="${NUM_EPISODES:-50}"
CHECKPOINT=""
FORWARD=()

usage() { awk 'NR>1 && !/^#/ {exit} NR>1 {sub(/^# ?/, ""); print}' "$0"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint|-c) CHECKPOINT="$2"; shift 2 ;;
        --num-envs|-n) NUM_ENVS="$2"; shift 2 ;;
        --num-episodes) NUM_EPISODES="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) FORWARD+=("$1"); shift ;;
    esac
done

MODELS_ROOT="$FLASHSAC_DIR/models/$GROUP"
if [ -z "$CHECKPOINT" ]; then
    # || true: without it, a no-match ls under `set -euo pipefail` kills the
    # script inside the $() before the error message below can ever print
    RUN_DIR=$(ls -td "$MODELS_ROOT/$EXP/$TRAIN_TASK"/*/ 2>/dev/null | head -1 || true)
    CHECKPOINT=$(ls -td "$RUN_DIR"/step*/ 2>/dev/null | head -1 || true)
fi
if [ ! -d "$CHECKPOINT" ]; then
    echo "error: no checkpoint found for GROUP=$GROUP EXP=$EXP TASK=$TRAIN_TASK under $MODELS_ROOT" >&2
    echo "hint: train that task/group first (and let it exit), or pass --checkpoint DIR" >&2
    exit 1
fi
echo "Playing checkpoint: $CHECKPOINT"
echo "Env: $PLAY_TASK ($NUM_ENVS envs, $NUM_EPISODES episodes)"

cd "$FLASHSAC_DIR"
export FLASH_RL_EXTRA_TASK_MODULES="${FLASH_RL_EXTRA_TASK_MODULES:-LEAP_Isaaclab.tasks}"
# hybrid-graphics laptop: render the Isaac Sim GUI on the NVIDIA dGPU
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

exec "$PYTHON" play_isaaclab.py \
    --checkpoint_path "$CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    --overrides env=isaaclab \
    --overrides "env.env_name=$PLAY_TASK" \
    --overrides gamma=0.99 \
    --overrides n_step=3 \
    --overrides agent.use_compile=false \
    --overrides agent.use_amp=false \
    --overrides agent.buffer_max_length=10000 \
    --overrides agent.buffer_min_length=10 \
    --overrides agent.buffer_device_type=cpu \
    ${FORWARD[@]+"${FORWARD[@]}"}
