#!/usr/bin/env bash
# Launch a kit-less training run on a provisioned AutoDL (seetacloud) box.
#
# Companion to scripts/launch/setup_kitless_remote.sh (run that first on a fresh
# instance). This script: rsyncs the repo (incremental -- carries local
# entrypoint changes over), makes sure the RL library is installed in the
# remote venv, then starts the run DETACHED (nohup) so a dropped SSH session
# does not kill it, and prints how to follow it.
#
# Usage:
#   KITLESS_SSH_PASS='...' scripts/launch/train_kitless_remote.sh -H <host> -p <port> \
#       [--task=Isaac-Humanoid-Direct-v0] [--entry=rl_games|rsl_rl] \
#       [--presets=newton_mjwarp|none] [--num-envs=N] [--max-iters=N] [--seed=N] \
#       [--checkpoint=/remote/path.pt] [--skip-sync]
#
# Defaults: task Isaac-Humanoid-Direct-v0, entry rl_games, presets newton_mjwarp.
# NOTE on resume: rl-games 1.x calls torch.load() without weights_only, which
# torch >= 2.6 refuses for its checkpoint format -- --checkpoint with rl_games
# may fail on this stack; fresh training is unaffected.
#
# Follow the run:
#   ssh -p <port> root@<host> tail -f <log printed below>
# Stop it:
#   ssh -p <port> root@<host> pkill -f train_kitless.py

set -euo pipefail

HOST="" PORT="" PASS="${KITLESS_SSH_PASS:-}"
TASK="Isaac-Humanoid-Direct-v0"
ENTRY="rl_games"          # rl_games | rsl_rl
PRESETS="newton_mjwarp"   # preset name, or 'none' for tasks without presets
NUM_ENVS="" MAX_ITERS="" SEED="" CHECKPOINT=""
SKIP_SYNC=0

usage() { sed -n '2,24p' "$0"; }
die() { echo "ERROR: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    -H) HOST="$2"; shift 2 ;;
    -p) PORT="$2"; shift 2 ;;
    -P) PASS="$2"; shift 2 ;;
    --task=*) TASK="${1#--task=}"; shift ;;
    --entry=*) ENTRY="${1#--entry=}"; shift ;;
    --presets=*) PRESETS="${1#--presets=}"; shift ;;
    --num-envs=*) NUM_ENVS="${1#--num-envs=}"; shift ;;
    --max-iters=*) MAX_ITERS="${1#--max-iters=}"; shift ;;
    --seed=*) SEED="${1#--seed=}"; shift ;;
    --checkpoint=*) CHECKPOINT="${1#--checkpoint=}"; shift ;;
    --skip-sync) SKIP_SYNC=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "error: unknown option $1"; usage; exit 1 ;;
  esac
done
[ -n "$HOST" ] || { usage; echo; echo "error: -H HOST and -p PORT are required"; exit 1; }
[ -n "$PORT" ] || { usage; echo; echo "error: -p PORT is required"; exit 1; }
[ "$ENTRY" = "rl_games" ] || [ "$ENTRY" = "rsl_rl" ] || die "--entry must be rl_games or rsl_rl"

if [ -n "$PASS" ]; then
  command -v sshpass >/dev/null || die "sshpass is required for password auth"
  SSH_BASE=(sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ConnectTimeout=20)
  RSYNC_BASE=(sshpass -p "$PASS" rsync -az --info=stats1 -e "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -p $PORT")
else
  SSH_BASE=(ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ConnectTimeout=20)
  RSYNC_BASE=(rsync -az --info=stats1 -e "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -p $PORT")
fi
DEST="root@$HOST"
rrun() { "${SSH_BASE[@]}" -p "$PORT" "$DEST" "export LC_ALL=C; $*"; }

REPO="$(git -C "$(dirname "$(readlink -f "$0")")/.." rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$REPO" ] || die "could not locate the local repo root (run from inside the repo)"
[ -d "$REPO/source/LEAP_Isaaclab" ] || die "$REPO does not look like the LEAP repo"

VENV="/root/isaaclab-kitless"
RREPO="/root/LEAP_Hand_Isaac_Lab"

# ---- 0. sanity --------------------------------------------------------------------------
rrun "test -x $VENV/bin/python" || die "no $VENV on the remote -- run scripts/launch/setup_kitless_remote.sh first"
rrun "nvidia-smi --query-gpu=memory.used --format=csv,noheader" | sed 's/^/GPU memory in use: /'

# ---- 1. sync ----------------------------------------------------------------------------
if [ "$SKIP_SYNC" = 0 ]; then
  echo "== rsyncing the repo (incremental) =="
  "${RSYNC_BASE[@]}" \
    --exclude logs/ --exclude .claude/ --exclude .serena/ --exclude .mcp.json \
    --exclude __pycache__ --exclude '*.pyc' \
    "$REPO/" "$DEST:$RREPO/" >/dev/null
fi

# ---- 2. RL library ----------------------------------------------------------------------
RL_SCRIPT="scripts/${ENTRY}/train_kitless.py"
rrun "test -f $RREPO/$RL_SCRIPT" || die "$RL_SCRIPT not found on the remote -- re-run without --skip-sync"
if [ "$ENTRY" = "rl_games" ]; then
  if ! rrun "$VENV/bin/python -c 'import rl_games'" 2>/dev/null; then
    echo "== installing rl-games into the remote venv =="
    rrun "$VENV/bin/pip install rl-games" || die "rl-games install failed"
  fi
  # isaaclab_rl's rl_games wrapper imports the classic gym package at module
  # level (compat shim for Denys88/rl_games#261); gymnasium does not provide it
  if ! rrun "$VENV/bin/python -c 'import gym'" 2>/dev/null; then
    echo "== installing classic gym (wrapper compat shim) =="
    rrun "$VENV/bin/pip install gym==0.26.2" || die "gym install failed"
  fi
fi

# ---- 3. launch detached ------------------------------------------------------------------
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RLOG="/root/train_${TASK}_${STAMP}.log"
ARGS="--task $TASK --headless presets=$PRESETS"
[ "$PRESETS" = "none" ] && ARGS="--task $TASK --headless"
[ -n "$NUM_ENVS" ] && ARGS="$ARGS --num_envs $NUM_ENVS"
[ -n "$MAX_ITERS" ] && ARGS="$ARGS --max_iterations $MAX_ITERS"
[ -n "$SEED" ] && ARGS="$ARGS --seed $SEED"
[ -n "$CHECKPOINT" ] && ARGS="$ARGS --checkpoint $CHECKPOINT"

echo "== launching: $VENV/bin/python $RL_SCRIPT $ARGS =="
rrun "cd $RREPO && nohup $VENV/bin/python $RL_SCRIPT $ARGS > $RLOG 2>&1 </dev/null & echo launched"

# ---- 4. first look -----------------------------------------------------------------------
sleep 20
echo; echo "== first log lines =="
rrun "head -30 $RLOG" || true
if rrun "grep -qE 'Traceback|Error' $RLOG"; then
  rrun "tail -20 $RLOG"
  die "the run crashed at startup -- see $DEST:$RLOG"
fi

cat <<EOF

Training is running on $DEST (survives SSH drops).

  follow:   ssh -p $PORT $DEST tail -f $RLOG
  stop:     ssh -p $PORT $DEST pkill -f train_kitless.py
  logs:     $RREPO/logs/rl_games/  (tensorboard: $RREPO/logs/rl_games/<algo-name>)

EOF
