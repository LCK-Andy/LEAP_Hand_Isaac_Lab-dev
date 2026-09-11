#!/usr/bin/env bash
# Provision kit-less Isaac Lab (no Isaac Sim / Kit) on a fresh AutoDL (seetacloud) GPU box.
#
# Replicates the setup verified 2026-09-11 on an AutoDL RTX 4090 D:
#   1. rsync this LEAP repo to /root/LEAP_Hand_Isaac_Lab (real LFS binaries, no logs)
#   2. venv at /root/isaaclab-kitless from /root/miniconda3 python + torch from the aliyun mirror
#   3. NVIDIA beta stack (isaaclab/newton/mujoco-warp/...) from local wheels:
#      the aliyun mirror AND pypi.org do NOT carry isaaclab 3.0.0b2; it lives on
#      pypi.nvidia.com, which AutoDL boxes usually cannot reach directly -- so the
#      wheels are downloaded on THIS machine (pip download, ~150 MB) and rsynced over.
#   4. usd-core (provides `pxr`, otherwise only shipped inside the isaacsim wheels)
#   5. symlink conda's too-old libstdc++ to the system one (omni.client needs GLIBCXX_3.4.30)
#   6. `pip install -e` the LEAP package
#   7. smoke: train_kitless on Isaac-Humanoid-v0 (newton_mjwarp), 5 iterations
#
# Every stage is idempotent -- re-running skips completed work, so the script is safe
# to re-launch after a network drop or an instance reboot.
#
# AutoDL notes:
#   - expects the standard image: root login, /root/miniconda3, aliyun pip mirror
#   - the stack (~10 GB) goes on the 30 GB system disk (/root), which IS included in
#     a saved image; /root/autodl-tmp is NOT -- keep the venv out of it
#   - /etc/network_turbo (github/huggingface proxy) is NOT needed by this script;
#     enable it manually only if you later git-pull on the box
#
# Usage:
#   KITLESS_SSH_PASS='...' scripts/setup_kitless_remote.sh -H connect.cqa1.seetacloud.com -p 30952
#   scripts/setup_kitless_remote.sh -H mybox.autodl.com -p 12345 --skip-smoke --skip-repo
#
# Requires locally: sshpass (when using a password), pip with internet access to
# pypi.nvidia.com (KITLESS_LOCAL_PIP overrides the interpreter, e.g. ~/env_isaaclab/bin/pip).

set -euo pipefail

# ---- pinned versions (match the local env_isaaclab install) --------------------------
ISAACLAB_VERSION=3.0.0b2.post1
NEWTON_VERSION=1.2.1
NEWTON_ACTUATORS_VERSION=0.1.0
NEWTON_USD_SCHEMAS_VERSION=0.2.0
MUJOCO_WARP_VERSION=3.8.0.3
WARP_VERSION=1.13.0
RSL_RL_VERSION=5.0.1
TORCH_VERSION=2.10.0
TORCHVISION_VERSION=0.25.0
SMOKE_TASK=Isaac-Humanoid-v0
SMOKE_PRESET=newton_mjwarp
SMOKE_ITERS=5

# ---- defaults ------------------------------------------------------------------------
HOST=""
PORT=""
PASS="${KITLESS_SSH_PASS:-}"
REPO="" # local repo; defaults to this script's repo root
RREPO="/root/LEAP_Hand_Isaac_Lab" # repo on the remote
RVENV="/root/isaaclab-kitless" # venv on the remote (must match the stage script below)
WHEELS_DIR="${KITLESS_WHEELS_DIR:-/tmp/leap_wheels}" # local wheel cache
SKIP_REPO=0
SKIP_SMOKE=0
MAX_WAIT_MIN="${KITLESS_MAX_WAIT_MIN:-45}"

usage() { sed -n '2,/^# Requires locally/p' "$0"; }

# long options first (getopts only sees short ones)
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-smoke) SKIP_SMOKE=1 ;;
    --skip-repo) SKIP_REPO=1 ;;
    --repo=*) REPO="${1#--repo=}" ;;
    -h | --help) usage; exit 0 ;;
    --*) echo "error: unknown option $1"; usage; exit 1 ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
[ ${#ARGS[@]} -eq 0 ] || set -- "${ARGS[@]}"
while getopts "H:p:P:h" opt; do
  case "$opt" in
    H) HOST="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    P) PASS="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done
[ -n "$HOST" ] || { usage; echo; echo "error: -H HOST is required"; exit 1; }
[ "$PORT" = "" ] && PORT=22

REPO="${REPO:-$(git -C "$(dirname "$(readlink -f "$0")")/.." rev-parse --show-toplevel 2>/dev/null || true)}"
[ -n "$REPO" ] || { echo "error: could not locate the local repo root (run from inside the repo, or pass --repo=/path/to/LEAP_Hand_Isaac_Lab)"; exit 1; }
[ -d "$REPO/source/LEAP_Isaaclab" ] || { echo "error: $REPO does not look like the LEAP repo"; exit 1; }

if [ -n "$PASS" ]; then
  command -v sshpass >/dev/null || { echo "error: sshpass is required for password auth (apt install sshpass)"; exit 1; }
  SSH_BASE=(sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ConnectTimeout=20)
  RSYNC_BASE=(sshpass -p "$PASS" rsync -az --info=stats1 -e "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -p $PORT")
else
  echo "note: KITLESS_SSH_PASS/-P not set, using key auth"
  SSH_BASE=(ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ConnectTimeout=20)
  RSYNC_BASE=(rsync -az --info=stats1 -e "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -p $PORT")
fi
DEST="root@$HOST"
rrun() { "${SSH_BASE[@]}" -p "$PORT" "$DEST" "export LC_ALL=C; $*"; }

stage() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

# ---- stage script that runs on the remote (idempotent) -------------------------------
TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT
STAGE_LOCAL="$TMPDIR_LOCAL/setup_kitless_stage.sh"
RLOG="/root/setup_kitless_stage.log"

cat >"$STAGE_LOCAL" <<'STAGE_EOF'
#!/bin/bash
set -uo pipefail
export LC_ALL=C
log() { echo "[stage] $*"; }
fail() { echo "FATAL: $*" >&2; exit 1; }

PY=/root/miniconda3/bin/python
VENV=/root/isaaclab-kitless
PIP=$VENV/bin/pip
WHEELS=/root/wheels
REPO=/root/LEAP_Hand_Isaac_Lab

# 1. venv
if [ ! -x "$PIP" ]; then
  log "creating venv at $VENV"
  "$PY" -m venv "$VENV" || fail "venv creation failed (is this the standard AutoDL image with miniconda3?)"
  "$PIP" install --upgrade pip wheel setuptools || fail "pip upgrade failed"
fi

# 2. torch from the aliyun mirror (pip no-ops when already satisfied)
log "installing torch (skips if present; a fresh box downloads ~3 GB)"
"$PIP" install torch==@TORCH@ torchvision==@TORCHVISION@ || fail "torch install failed"
log "TORCH_OK"

# 3. NVIDIA kit-less stack from the rsynced wheels
if ! ls "$WHEELS"/isaaclab-*.whl >/dev/null 2>&1; then
  echo "WHEELS_MISSING"
  exit 42
fi
log "installing isaaclab/newton/mujoco-warp from $WHEELS (deps resolve from the aliyun mirror)"
"$PIP" install "$WHEELS"/*.whl || fail "kit-less stack install failed"
log "STACK_OK"

# 4. pxr bindings -- without the isaacsim wheels nothing provides them
log "installing usd-core"
"$PIP" install usd-core || fail "usd-core install failed"

# 5. conda's libstdc++ is older than the system one and omni.client needs GLIBCXX_3.4.30
SYS_LIB=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
CONDA_LIB="$(dirname "$(dirname "$PY")")/lib/libstdc++.so.6"
if [ -e "$CONDA_LIB" ] && [ -e "$SYS_LIB" ] && [ "$(readlink -f "$CONDA_LIB")" != "$(readlink -f "$SYS_LIB")" ]; then
  if command -v strings >/dev/null 2>&1 && strings "$CONDA_LIB" | grep -q GLIBCXX_3.4.30; then
    log "conda libstdc++ already new enough"
  else
    cp -n "$CONDA_LIB" "$CONDA_LIB.bak"
    ln -sfn "$SYS_LIB" "$CONDA_LIB"
    log "symlinked $CONDA_LIB -> $SYS_LIB (backup: $CONDA_LIB.bak)"
  fi
fi

# 6. LEAP package
log "installing the LEAP package (editable)"
"$PIP" install -e "$REPO/source/LEAP_Isaaclab" || fail "editable install failed"

echo "STACK_DONE"
STAGE_EOF

sed -e "s|@TORCH@|$TORCH_VERSION|g" -e "s|@TORCHVISION@|$TORCHVISION_VERSION|g" -i "$STAGE_LOCAL"

# ---- 1. probe ------------------------------------------------------------------------
stage "probing $DEST:$PORT"
rrun 'echo "login OK"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader' \
  || die "cannot SSH in (check host/port/password)"
GPU_FREE_MB=$(rrun 'df -m / | awk "NR==2 {print \$4}"')
[ "${GPU_FREE_MB:-0}" -ge 15360 ] || die "only ${GPU_FREE_MB:-0} MB free on / -- the stack needs ~15 GB"
rrun '/root/miniconda3/bin/python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)"' \
  || die "/root/miniconda3/bin/python missing or not 3.10-3.12 -- not the standard AutoDL image?"

# ---- 2. repo rsync -------------------------------------------------------------------
if [ "$SKIP_REPO" = 0 ]; then
  stage "rsyncing the repo to $DEST:$RREPO (first run ~330 MB, re-runs are incremental)"
  "${RSYNC_BASE[@]}" \
    --exclude logs/ --exclude .claude/ --exclude .serena/ --exclude .mcp.json \
    --exclude __pycache__ --exclude '*.pyc' \
    "$REPO/" "$DEST:$RREPO/" || die "repo rsync failed"
else
  stage "skipping repo rsync"
fi

# ---- 3. wheels -----------------------------------------------------------------------
stage "ensuring the NVIDIA beta wheels are on the remote"
have_remote_wheels=$(rrun 'ls /root/wheels/isaaclab-*.whl >/dev/null 2>&1 && echo yes || echo no')
WHEEL_SPECS=(
  "isaaclab==$ISAACLAB_VERSION" "newton==$NEWTON_VERSION" "mujoco-warp==$MUJOCO_WARP_VERSION"
  "newton-actuators==$NEWTON_ACTUATORS_VERSION" "newton-usd-schemas==$NEWTON_USD_SCHEMAS_VERSION"
  "warp-lang==$WARP_VERSION" "rsl-rl-lib==$RSL_RL_VERSION"
)
if [ "$have_remote_wheels" = yes ]; then
  echo "remote /root/wheels already populated"
else
  if ! ls "$WHEELS_DIR"/isaaclab-*.whl >/dev/null 2>&1; then
    PIP_BIN="${KITLESS_LOCAL_PIP:-}"
    if [ -z "$PIP_BIN" ]; then
      for c in pip3 pip "$HOME/env_isaaclab/bin/pip"; do
        if command -v "$c" >/dev/null 2>&1; then PIP_BIN="$c"; break; fi
      done
    fi
    [ -n "$PIP_BIN" ] || die "no local pip found to download wheels (set KITLESS_LOCAL_PIP)"
    echo "downloading wheels locally with $PIP_BIN (~150 MB from pypi.nvidia.com)"
    mkdir -p "$WHEELS_DIR"
    "$PIP_BIN" download --no-deps --pre -d "$WHEELS_DIR" \
      --index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple \
      "${WHEEL_SPECS[@]}" || die "local wheel download failed -- is pypi.nvidia.com reachable from this machine?"
    ls "$WHEELS_DIR"/isaaclab-*.whl >/dev/null || die "wheel download produced no isaaclab wheel"
  fi
  "${RSYNC_BASE[@]}" "$WHEELS_DIR/" "$DEST:/root/wheels/" || die "wheel rsync failed"
fi

# ---- 4. run the stage script (detached; survives SSH drops) --------------------------
stage "installing the stack (fresh box: ~15-40 min, mostly the torch download)"
"${RSYNC_BASE[@]}" "$STAGE_LOCAL" "$DEST:/root/setup_kitless_stage.sh" || die "stage script upload failed"
rrun "nohup bash /root/setup_kitless_stage.sh > $RLOG 2>&1 & echo started"

last=""
for _ in $(seq 1 $((MAX_WAIT_MIN * 3))); do
  sleep 20
  new=$(rrun "grep -aE '^\[stage\]|FATAL|WHEELS_MISSING|STACK_DONE' $RLOG | tail -5" || true)
  if [ "$new" != "$last" ]; then
    echo "$new"
    last="$new"
  fi
  case "$new" in
    *STACK_DONE*) break ;;
    *FATAL*|*WHEELS_MISSING*) die "remote setup failed -- see $DEST:$RLOG" ;;
  esac
  # [s] trick so this poll command's own cmdline (which contains the pattern) can't self-match
  if ! rrun 'pgrep -f "[s]etup_kitless_stage.sh" >/dev/null'; then
    die "stage process died without a marker -- see $DEST:$RLOG"
  fi
done
rrun "grep -aq STACK_DONE $RLOG" || die "timed out after ${MAX_WAIT_MIN} min (still installing?) -- re-run to resume"

# ---- 5. smoke ------------------------------------------------------------------------
if [ "$SKIP_SMOKE" = 0 ]; then
  stage "smoke: $SMOKE_TASK on $SMOKE_PRESET, $SMOKE_ITERS iterations (first run compiles warp kernels)"
  rc=$(rrun "cd $RREPO && timeout 1500 $RVENV/bin/python scripts/rsl_rl/train_kitless.py \
    --task $SMOKE_TASK --headless presets=$SMOKE_PRESET --max_iterations $SMOKE_ITERS \
    > /root/smoke_kitless.log 2>&1; echo \$?")
  if [ "$rc" != 0 ]; then
    rrun "tail -30 /root/smoke_kitless.log"
    die "smoke run exited with rc=$rc"
  fi
  rrun "ls $RREPO/logs/rsl_rl/humanoid/*/model_$((SMOKE_ITERS - 1)).pt >/dev/null" \
    || die "smoke ran but no checkpoint was saved"
  rrun "grep -aE 'Mean reward|Iteration time' /root/smoke_kitless.log | tail -4"
else
  stage "skipping smoke"
fi

# ---- summary -------------------------------------------------------------------------
stage "done"
rrun "$RVENV/bin/pip list 2>/dev/null | grep -iE '^(isaaclab|newton|mujoco-warp|warp-lang|torch|rsl-rl-lib|usd-core|LEAP)' || true"
cat <<SUMMARY

Kit-less Isaac Lab is ready on $DEST. To train there:

  ssh -p $PORT $DEST
  cd $RREPO
  /root/isaaclab-kitless/bin/python scripts/rsl_rl/train_kitless.py \\
      --task $SMOKE_TASK --headless presets=$SMOKE_PRESET

Only tasks with a kit-less physics preset (e.g. presets=$SMOKE_PRESET) can run on
this box; PhysX/Kit-rendered tasks (the LEAP repose/direct set) need Isaac Sim.
SUMMARY
