#!/usr/bin/env bash
# Play a FlashSAC-trained policy for Isaac-Repose-Cube-LEAP-Direct-v0 with the Isaac Sim Kit GUI.
#
# The Isaac Sim window always opens (play_isaaclab.py runs headless=False); there is no
# --viz flag. Unknown args are forwarded verbatim to play_isaaclab.py, which only
# supports: --config_path, --config_name, --overrides KEY=VAL, --checkpoint_path,
# --num_envs, --num_episodes.
#
# Usage:
#   scripts/flashsac_play_leap_direct.sh                                 # newest ckpt, 16 envs, 50 episodes
#   scripts/flashsac_play_leap_direct.sh --num-envs 4 --num-episodes 200
#   scripts/flashsac_play_leap_direct.sh --checkpoint models/.../step49152
#   scripts/flashsac_play_leap_direct.sh --overrides seed=1000
#
# Options:
#   --checkpoint, -c DIR   Checkpoint dir (default: newest step*/ of the newest run
#                          under ~/Documents/projects/FlashSAC/models/leap_direct/)
#   --num-envs, -n N       Parallel envs in the grid (default 16)
#   --num-episodes N       Episodes before exit (default 50; env var NUM_EPISODES)
#   -h, --help             This help

set -euo pipefail

FLASHSAC_DIR="${FLASHSAC_DIR:-$HOME/Documents/projects/FlashSAC}"
PYTHON="${PYTHON:-$HOME/env_isaaclab/bin/python}"
TASK="${TASK:-Isaac-Repose-Cube-LEAP-Direct-v0}"
NUM_ENVS="${NUM_ENVS:-16}"
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

MODELS_ROOT="$FLASHSAC_DIR/models/leap_direct"
if [ -z "$CHECKPOINT" ]; then
    RUN_DIR=$(ls -td "$MODELS_ROOT"/*/"$TASK"/*/ 2>/dev/null | head -1)
    CHECKPOINT=$(ls -td "$RUN_DIR"/step*/ 2>/dev/null | head -1)
fi
if [ ! -d "$CHECKPOINT" ]; then
    echo "error: checkpoint dir not found: $CHECKPOINT" >&2
    exit 1
fi
echo "Playing checkpoint: $CHECKPOINT"

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
    --overrides "env.env_name=$TASK" \
    --overrides gamma=0.99 \
    --overrides n_step=3 \
    --overrides agent.use_compile=false \
    --overrides agent.use_amp=false \
    --overrides agent.buffer_max_length=10000 \
    --overrides agent.buffer_min_length=10 \
    --overrides agent.buffer_device_type=cpu \
    ${FORWARD[@]+"${FORWARD[@]}"}
