#!/usr/bin/env bash
# Provision Isaac Lab on a fresh AutoDL (seetacloud) GPU box: kit-less (default)
# or FULL Isaac Sim (--full). Replicates setups verified 2026-09-11/12 on an
# AutoDL RTX 4090 D container (Chongqing).
#
# Network reality on AutoDL boxes (measured 2026-09-11) -- baked into this script:
#   - pypi.org and download.pytorch.org are UNREACHABLE from the container, and
#     the image may ship no pip config at all (or the user may have set the
#     broken "official" one). The script therefore configures indexes EXPLICITLY:
#     tuna mirror for PyPI (16 MB/s) + pypi.nvidia.com extra (12-18 MB/s).
#     The aliyun pypi mirror measured 0.13 MB/s -- do not use it.
#   - pypi.nvidia.com IS directly reachable (the old "must relay via laptop"
#     assumption is obsolete); isaacsim wheels pull fine from the box.
#   - torch is the exception: isaaclab's installer pins the exact local tag
#     2.10.0+cu128, which exists only on download.pytorch.org. In --full mode
#     those wheels (~880 MB) are downloaded on THIS machine and rsynced over;
#     the nvidia-*-cu12 deps resolve from tuna.
#   - /etc/network_turbo (github/huggingface proxy) must NOT be sourced
#     globally: it breaks the pip indexes ("from versions: none"). github is
#     reachable only intermittently from AutoDL, so git traffic to github.com
#     is routed through the proxy via `git config http.proxy` instead, while
#     pip stays direct (observed 2026-09-12: direct github died with GnuTLS
#     -110 mid-clone; pip behind the proxy could not resolve setuptools).
#   - pip streams downloads to $TMPDIR: the default /tmp sits on the 30 GB
#     system disk, and isaacsim-extscache-kit alone is a single 5.9 GB wheel
#     that filled it (OSError Errno 28). TMPDIR/PIP_CACHE_DIR are pinned to the
#     data disk whenever /root/autodl-tmp exists.
#
# kit-less mode (default):
#   1. rsync this LEAP repo to /root/LEAP_Hand_Isaac_Lab (real LFS binaries, no logs)
#   2. venv /root/isaaclab-kitless from /root/miniconda3 python (3.10-3.12 ok)
#   3. torch (plain PyPI build, no +cu128 tag needed here) from tuna
#   4. NVIDIA beta stack (isaaclab/newton/mujoco-warp/...) from local wheels:
#      rsynced from pypi.nvidia.com via this machine (~150 MB, proven path)
#   5. usd-core (pxr bindings); libstdc++ symlink; tmux; LEAP package editable
#   6. smoke: train_kitless on Isaac-Humanoid-v0 (newton_mjwarp), 5 iterations
#   The stack (~10 GB) stays on the 30 GB system disk (/root), which IS included
#   in a saved AutoDL image; /root/autodl-tmp is NOT -- keep the venv out of it.
#
# --full mode (Isaac Sim 6.0.1.0 + PhysX, as verified 2026-09-11):
#   1. /root/autodl-tmp (50 GB data disk) is REQUIRED -- the full stack is
#      ~35-40 GB and does not fit the system disk. Needs miniconda python 3.12
#      (Isaac Lab 3.0 requires >=3.12, unlike kit-less which accepts 3.10+).
#   2. venv /root/autodl-tmp/env_isaaclab
#   3. shallow clone IsaacLab release/3.0.0-beta2 -> /root/autodl-tmp/IsaacLab
#   4. isaacsim[all,extscache]==6.0.1.0 (~20 GB) from tuna + nvidia indexes
#   5. torch 2.10.0+cu128 from the laptop-relayed wheels (deps from tuna)
#   6. ./isaaclab.sh --install (no-ops on isaacsim/torch already satisfied)
#   7. headless fixups: EULA marker in isaacsim/kit/ (NOT the package root --
#      that one is silently ignored; OMNI_KIT_ACCEPT_EULA=YES also works) and
#      apt libxt6/libsm6/libice6 for the MaterialX extensions
#   8. smoke: Isaac-Ant-Direct-v0 on newton_mjwarp AND on physx --headless
#   9. pip cache purge + relay-wheel cleanup (leaves headroom on the data disk)
#
# Every stage is idempotent -- re-running skips completed work, so the script is
# safe to re-launch after a network drop or an instance reboot.
#
# Usage:
#   KITLESS_SSH_PASS='...' scripts/setup_kitless_remote.sh -H connect.cqa1.seetacloud.com -p 30952
#   scripts/setup_kitless_remote.sh -H mybox.autodl.com -p 12345 --skip-smoke --skip-repo
#   scripts/setup_kitless_remote.sh -H mybox.autodl.com -p 12345 --full
#
# Requires locally: sshpass (when using a password), pip with internet access to
# pypi.nvidia.com (kit-less) and download.pytorch.org (--full); KITLESS_LOCAL_PIP
# overrides the interpreter, e.g. ~/env_isaaclab/bin/pip.

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
ISAACSIM_VERSION=6.0.1.0
ISAACLAB_BRANCH=release/3.0.0-beta2
TORCH_CU128_VERSION=2.10.0+cu128
TORCHVISION_CU128_VERSION=0.25.0+cu128
TUNA_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
NVIDIA_INDEX=https://pypi.nvidia.com
SMOKE_TASK=Isaac-Humanoid-v0
SMOKE_PRESET=newton_mjwarp
SMOKE_ITERS=5
FULL_SMOKE_TASK=Isaac-Ant-Direct-v0

# ---- defaults ------------------------------------------------------------------------
HOST=""
PORT=""
PASS="${KITLESS_SSH_PASS:-}"
REPO="" # local repo; defaults to this script's repo root
RREPO="/root/LEAP_Hand_Isaac_Lab" # repo on the remote (kit-less mode)
RVENV="/root/isaaclab-kitless" # venv on the remote (kit-less; must match train_kitless_remote.sh)
FROOT="/root/autodl-tmp" # data-disk root (--full mode)
FVENV="$FROOT/env_isaaclab" # venv on the remote (--full mode)
FISAACLAB="$FROOT/IsaacLab" # IsaacLab clone (--full mode)
WHEELS_DIR="${KITLESS_WHEELS_DIR:-/tmp/leap_wheels}" # local wheel cache
SKIP_REPO=0
SKIP_SMOKE=0
FULL=0
MAX_WAIT_MIN="${KITLESS_MAX_WAIT_MIN:-45}"

usage() { sed -n '2,/^# Requires locally/p' "$0"; }

# long options first (getopts only sees short ones)
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-smoke) SKIP_SMOKE=1 ;;
    --skip-repo) SKIP_REPO=1 ;;
    --full) FULL=1 ;;
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
# the full install streams ~20 GB and builds imgui from source: allow 90 min
if [ "$FULL" = 1 ] && [ "$MAX_WAIT_MIN" = 45 ]; then MAX_WAIT_MIN=90; fi

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

# ---- stage scripts that run on the remote (idempotent) -------------------------------
TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT
STAGE_LOCAL="$TMPDIR_LOCAL/setup_stage.sh"
if [ "$FULL" = 1 ]; then RLOG="/root/setup_full_stage.log"; else RLOG="/root/setup_kitless_stage.log"; fi

# Shared prologue for both stage scripts: explicit pip indexes + TMPDIR guard.
# Written as a function-style snippet and substituted into both heredocs below.
stage_prologue='
# Explicit index config: fresh venvs inherit whatever (possibly broken) global
# config the image or the user left behind; pypi.org itself is unreachable from
# AutoDL boxes, so pin tuna (fast PyPI mirror) + pypi.nvidia.com (NVIDIA stack).
PIP_BIN_FOR_CFG="$PIP"; [ -x "$PIP" ] || PIP_BIN_FOR_CFG=/root/miniconda3/bin/pip
"$PIP_BIN_FOR_CFG" config set global.index-url @TUNA@ >/dev/null || fail "pip config set index-url failed"
"$PIP_BIN_FOR_CFG" config set global.extra-index-url @NVIDIA@ >/dev/null || fail "pip config set extra-index-url failed"
# pip streams downloads to $TMPDIR: default /tmp lives on the small system disk
# and multi-GB wheels fill it (Errno 28). Keep temp + cache on the data disk.
if [ -d /root/autodl-tmp ]; then
  mkdir -p /root/autodl-tmp/tmp /root/autodl-tmp/.cache/pip
  export TMPDIR=/root/autodl-tmp/tmp
  export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
fi
'

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

# 1. venv (fresh venv pip has no index config yet -> pass tuna explicitly)
if [ ! -x "$PIP" ]; then
  log "creating venv at $VENV"
  "$PY" -m venv "$VENV" || fail "venv creation failed (is this the standard AutoDL image with miniconda3?)"
  "$PIP" install --upgrade pip wheel -i @TUNA@ || fail "pip upgrade failed"
fi
@PROLOGUE@

# 2. torch (plain PyPI build; the +cu128 local tag is only required by
#    isaaclab.sh's installer in --full mode, not by the kit-less stack)
log "installing torch (skips if present; a fresh box downloads ~3 GB)"
"$PIP" install torch==@TORCH@ torchvision==@TORCHVISION@ || fail "torch install failed"
log "TORCH_OK"

# 3. NVIDIA kit-less stack from the rsynced wheels
if ! ls "$WHEELS"/isaaclab-*.whl >/dev/null 2>&1; then
  echo "WHEELS_MISSING"
  exit 42
fi
log "installing isaaclab/newton/mujoco-warp from $WHEELS (deps resolve from tuna)"
"$PIP" install "$WHEELS"/*.whl || fail "kit-less stack install failed"
log "STACK_OK"

# 4. pxr bindings -- without the isaacsim wheels nothing provides them
log "installing usd-core"
"$PIP" install usd-core || fail "usd-core install failed"

# 5. conda's libstdc++ is older than the system one and omni.client needs GLIBCXX_3.4.30
SYS_LIB=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
CONDA_LIB="$(dirname "$(dirname "$PY")")/lib/libstdc++.so.6"
if [ -e "$CONDA_LIB" ] && [ -e "$SYS_LIB" ] && [ "$(readlink -f "$CONDA_LIB")" != "$(readlink -f "$SYS_LIB")" ]; then
  # grep without -q: reads all input, so strings never SIGPIPEs under pipefail
  if command -v strings >/dev/null 2>&1 && strings "$CONDA_LIB" 2>/dev/null | grep GLIBCXX_3.4.30 >/dev/null; then
    log "conda libstdc++ already new enough"
  else
    cp -n "$CONDA_LIB" "$CONDA_LIB.bak"
    ln -sfn "$SYS_LIB" "$CONDA_LIB"
    log "symlinked $CONDA_LIB -> $SYS_LIB (backup: $CONDA_LIB.bak)"
  fi
fi

# 6. tmux -- the base image ships none; detached training runs live in tmux sessions
if command -v tmux >/dev/null 2>&1; then
  log "tmux already present ($(tmux -V))"
elif apt-get update -qq && apt-get install -y -qq tmux; then
  log "installed $(tmux -V) via apt"
else
  log "WARNING: tmux install failed -- runs can still be launched with nohup"
fi

# 7. LEAP package
log "installing the LEAP package (editable)"
"$PIP" install -e "$REPO/source/LEAP_Isaaclab" || fail "editable install failed"

echo "STACK_DONE"
STAGE_EOF

if [ "$FULL" = 1 ]; then
  cat >"$STAGE_LOCAL" <<'FULL_EOF'
#!/bin/bash
set -uo pipefail
export LC_ALL=C
log() { echo "[stage] $*"; }
fail() { echo "FATAL: $*" >&2; exit 1; }

ROOT=/root/autodl-tmp
VENV=$ROOT/env_isaaclab
PIP=$VENV/bin/pip
ISAACLAB=$ROOT/IsaacLab

# 1. venv -- Isaac Lab 3.0 needs python 3.12 exactly here (kit-less allows 3.10+)
if [ ! -x "$PIP" ]; then
  log "creating venv at $VENV"
  /root/miniconda3/bin/python3.12 -m venv "$VENV" || fail "venv creation failed (need /root/miniconda3 python 3.12)"
  "$PIP" install --upgrade pip wheel -i @TUNA@ || fail "pip upgrade failed"
fi
@PROLOGUE@

# 2. Isaac Lab repo (shallow). github needs the AutoDL proxy; turbo slows the
#    pip indexes, so source it ONLY inside the clone subshell.
if [ ! -f "$ISAACLAB/isaaclab.sh" ]; then
  log "cloning IsaacLab @BRANCH@ -> $ISAACLAB"
  ( source /etc/network_turbo >/dev/null 2>&1 || true
    git clone --depth 1 --branch @BRANCH@ https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB" ) \
    || fail "IsaacLab clone failed (is /etc/network_turbo available?)"
else
  log "IsaacLab clone already present"
fi

# 3. Isaac Sim full stack (~20 GB). Single wheels reach 5.9 GB -- TMPDIR must be
#    on this data disk (see prologue) or the system disk fills mid-download.
if "$PIP" show isaacsim >/dev/null 2>&1; then
  log "isaacsim already installed"
else
  log "installing isaacsim[all,extscache]==@ISAACSIM@ (~20 GB; 20-40 min at 12-18 MB/s)"
  "$PIP" install --timeout 25 --retries 8 "isaacsim[all,extscache]==@ISAACSIM@" \
    || fail "isaacsim install failed (was TMPDIR on the data disk? Errno 28 means it was not)"
fi

# 4. torch with the EXACT local tag isaaclab's installer demands -- only
#    download.pytorch.org hosts it (unreachable from AutoDL), so it comes from
#    the laptop-relayed wheels; the nvidia-*-cu12 deps resolve from tuna.
#    NB: no `grep -q` in a pipe here -- with pipefail, grep -q's early exit
#    SIGPIPEs pip and the check fails even when torch IS installed.
torch_ver=$("$PIP" show torch 2>/dev/null | sed -n 's/^Version: //p')
if [ "$torch_ver" = "@TORCHCU@" ]; then
  log "torch @TORCHCU@ already installed"
else
  ls /root/wheels/torch-@TORCHCU@-*.whl >/dev/null 2>&1 || { echo "WHEELS_MISSING"; exit 42; }
  "$PIP" install /root/wheels/torch-@TORCHCU@-*.whl /root/wheels/torchvision-@TORCHVISIONCU@-*.whl \
    || fail "torch+cu128 install failed"
fi

# 5. Isaac Lab itself. The installer git-clones deps from github (rl_games,
#    robomimic, ...) AND pip-installs from tuna/nvidia -- two conflicting
#    network needs: github access from AutoDL is intermittent (GnuTLS -110
#    mid-clone), while the pip indexes BREAK behind the network_turbo proxy
#    ("from versions: none"; both observed 2026-09-12). Split them: route ONLY
#    github.com git traffic through the proxy, keep pip direct.
GH_PROXY=$(source /etc/network_turbo >/dev/null 2>&1; printf %s "$https_proxy")
if [ -n "$GH_PROXY" ]; then
  git config --global http.https://github.com.proxy "$GH_PROXY"
  log "github.com git traffic routed via $GH_PROXY (pip stays direct)"
else
  log "WARNING: no network_turbo proxy found -- github clones go direct and may flake"
fi
log "running isaaclab.sh --install (idempotent; fresh box ~15-20 min)"
# retry once after a cache purge: a single corrupted/poisoned http-cache entry
# (e.g. written while a proxy mangled a response) makes every later resolve of
# that package fail with "from versions: none" -- observed 2026-09-12
( cd "$ISAACLAB" && source "$VENV/bin/activate" && ./isaaclab.sh --install ) \
  || { log "install failed -- purging pip cache and retrying once"; "$PIP" cache purge >/dev/null 2>&1 || true
       ( cd "$ISAACLAB" && source "$VENV/bin/activate" && ./isaaclab.sh --install ) ; } \
  || fail "isaaclab.sh --install failed (github proxy reachable? pip indexes reachable?)"

# 6. headless-container fixups:
#    - EULA: non-interactive shells EOF on the prompt and die. The marker file
#      must live in isaacsim/kit/ -- a marker in the isaacsim package root is
#      silently ignored (verified the hard way).
SP_DIR="$("$PIP" show isaacsim 2>/dev/null | sed -n 's/^Location: //p')/isaacsim"
if [ -d "$SP_DIR/kit" ]; then
  echo yes > "$SP_DIR/kit/EULA_ACCEPTED" && log "EULA accepted (marker in isaacsim/kit/)"
else
  log "WARNING: could not write EULA marker -- export OMNI_KIT_ACCEPT_EULA=YES before running"
fi
#    - MaterialX extensions fail to load without libXt (physics unaffected)
apt-get install -y -qq libxt6 libsm6 libice6 >/dev/null 2>&1 \
  || log "WARNING: X libs install failed (MaterialX stays broken; physics fine)"

# 7. reclaim data-disk space: the pip cache holds ~10 GB of transient wheels
"$PIP" cache purge >/dev/null 2>&1 || true
rm -rf /root/wheels/torch-*.whl /root/wheels/torchvision-*.whl 2>/dev/null || true
rm -rf "$ROOT/tmp"/* 2>/dev/null || true

echo "STACK_DONE"
FULL_EOF
fi

# ---- substitute the shared prologue + version pins into the chosen stage script -------
PROLOGUE_FILE="$TMPDIR_LOCAL/prologue.txt"
printf '%s' "$stage_prologue" | sed -e "s|@TUNA@|$TUNA_INDEX|g" -e "s|@NVIDIA@|$NVIDIA_INDEX|g" > "$PROLOGUE_FILE"
python3 - "$STAGE_LOCAL" "$PROLOGUE_FILE" <<'PYEOF'
import sys
stage_path, prologue_path = sys.argv[1], sys.argv[2]
with open(stage_path) as f: stage = f.read()
with open(prologue_path) as f: prologue = f.read()
stage = stage.replace("@PROLOGUE@", prologue)
with open(stage_path, "w") as f: f.write(stage)
PYEOF
sed -e "s|@TUNA@|$TUNA_INDEX|g" -e "s|@NVIDIA@|$NVIDIA_INDEX|g" \
    -e "s|@TORCH@|$TORCH_VERSION|g" -e "s|@TORCHVISION@|$TORCHVISION_VERSION|g" \
    -e "s|@ISAACSIM@|$ISAACSIM_VERSION|g" -e "s|@BRANCH@|$ISAACLAB_BRANCH|g" \
    -e "s|@TORCHCU@|$TORCH_CU128_VERSION|g" -e "s|@TORCHVISIONCU@|$TORCHVISION_CU128_VERSION|g" \
    -i "$STAGE_LOCAL"
rm -f "$PROLOGUE_FILE"

# ---- 1. probe ------------------------------------------------------------------------
stage "probing $DEST:$PORT (mode: $([ "$FULL" = 1 ] && echo FULL || echo kit-less))"
rrun 'echo "login OK"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader' \
  || die "cannot SSH in (check host/port/password)"
if [ "$FULL" = 1 ]; then
  rrun 'test -d /root/autodl-tmp' || die "--full needs /root/autodl-tmp (the AutoDL data disk) -- the stack (~35-40 GB) does not fit the 30 GB system disk"
  # the ~45 GB free-space gate only applies to a FRESH provision; re-running on
  # an already-provisioned box (e.g. to repair) must not be blocked by the very
  # disk usage the install created
  already_full=$(rrun "$FVENV/bin/pip show isaacsim >/dev/null 2>&1 && echo yes || echo no")
  if [ "$already_full" != yes ]; then
    DATA_FREE_MB=$(rrun 'df -m /root/autodl-tmp | awk "NR==2 {print \$4}"')
    [ "${DATA_FREE_MB:-0}" -ge 45000 ] || die "only ${DATA_FREE_MB:-0} MB free on /root/autodl-tmp -- the full install peaks at ~45 GB (stack + transient wheel cache)"
  fi
  rrun '/root/miniconda3/bin/python3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)"' \
    || die "/root/miniconda3 python3.12 missing -- Isaac Lab 3.0 requires 3.12 in --full mode"
else
  ROOT_FREE_MB=$(rrun 'df -m / | awk "NR==2 {print \$4}"')
  [ "${ROOT_FREE_MB:-0}" -ge 15360 ] || die "only ${ROOT_FREE_MB:-0} MB free on / -- the kit-less stack needs ~15 GB"
  rrun '/root/miniconda3/bin/python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)"' \
    || die "/root/miniconda3/bin/python missing or not 3.10-3.12 -- not the standard AutoDL image?"
fi

# ---- 2. repo rsync (kit-less only) ----------------------------------------------------
if [ "$FULL" = 1 ]; then
  stage "skipping repo rsync (--full does not need the LEAP repo)"
elif [ "$SKIP_REPO" = 0 ]; then
  stage "rsyncing the repo to $DEST:$RREPO (first run ~330 MB, re-runs are incremental)"
  "${RSYNC_BASE[@]}" \
    --exclude logs/ --exclude .claude/ --exclude .serena/ --exclude .mcp.json \
    --exclude __pycache__ --exclude '*.pyc' \
    "$REPO/" "$DEST:$RREPO/" || die "repo rsync failed"
else
  stage "skipping repo rsync"
fi

# ---- 3. wheels ------------------------------------------------------------------------
stage "ensuring wheels are on the remote"
if [ "$FULL" = 1 ]; then
  # torch+cu128 relay: the ONLY package domestic indexes cannot serve
  have_torch=$(rrun "$FVENV/bin/pip show torch 2>/dev/null | grep -q 'Version: $TORCH_CU128_VERSION' && echo yes || echo no")
  if [ "$have_torch" = yes ]; then
    echo "remote already has torch $TORCH_CU128_VERSION"
  else
    if ! ls "$WHEELS_DIR"/torch-2.10.0+cu128-*.whl >/dev/null 2>&1; then
      PIP_BIN="${KITLESS_LOCAL_PIP:-}"
      if [ -z "$PIP_BIN" ]; then
        for c in pip3 pip "$HOME/env_isaaclab/bin/pip"; do
          if command -v "$c" >/dev/null 2>&1; then PIP_BIN="$c"; break; fi
        done
      fi
      [ -n "$PIP_BIN" ] || die "no local pip found to download the torch wheels (set KITLESS_LOCAL_PIP)"
      echo "downloading torch $TORCH_CU128_VERSION locally with $PIP_BIN (~880 MB from download.pytorch.org)"
      mkdir -p "$WHEELS_DIR"
      "$PIP_BIN" download --no-deps -d "$WHEELS_DIR" \
        --index-url https://download.pytorch.org/whl/cu128 \
        "torch==$TORCH_CU128_VERSION" "torchvision==$TORCHVISION_CU128_VERSION" \
        || die "local torch wheel download failed -- is download.pytorch.org reachable from this machine?"
      ls "$WHEELS_DIR"/torch-2.10.0+cu128-*.whl >/dev/null || die "wheel download produced no torch+cu128 wheel"
    fi
    "${RSYNC_BASE[@]}" "$WHEELS_DIR"/torch-*+cu128-*.whl "$DEST:/root/wheels/" || die "torch wheel rsync failed"
  fi
else
  # kit-less NVIDIA beta stack (~150 MB); deps resolve on the remote from tuna
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
fi

# ---- 4. run the stage script (detached; survives SSH drops) --------------------------
stage "installing the stack (fresh box: kit-less ~15-40 min, --full ~40-90 min)"
"${RSYNC_BASE[@]}" "$STAGE_LOCAL" "$DEST:/root/setup_stage.sh" || die "stage script upload failed"
rrun "nohup bash /root/setup_stage.sh > $RLOG 2>&1 & echo started"

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
  if ! rrun 'pgrep -f "[s]etup_stage.sh" >/dev/null'; then
    die "stage process died without a marker -- see $DEST:$RLOG"
  fi
done
rrun "grep -aq STACK_DONE $RLOG" || die "timed out after ${MAX_WAIT_MIN} min (still installing?) -- re-run to resume"

# ---- 5. smoke ------------------------------------------------------------------------
if [ "$SKIP_SMOKE" = 0 ] && [ "$FULL" = 1 ]; then
  stage "smoke (full): $FULL_SMOKE_TASK on newton_mjwarp, 3 iterations"
  rc=$(rrun "cd $FISAACLAB && timeout 900 $FVENV/bin/python scripts/reinforcement_learning/train.py \
    --rl_library rsl_rl --task=$FULL_SMOKE_TASK --num_envs=512 --max_iterations=3 physics=newton_mjwarp \
    > /root/smoke_full_newton.log 2>&1; echo \$?")
  [ "$rc" = 0 ] || { rrun "tail -30 /root/smoke_full_newton.log"; die "newton smoke rc=$rc"; }
  rrun "grep -aE 'Steps per second|Training time' /root/smoke_full_newton.log | tail -3"

  stage "smoke (full): $FULL_SMOKE_TASK on physx (full Isaac Sim, headless), 2 iterations"
  rc=$(rrun "cd $FISAACLAB && OMNI_KIT_ACCEPT_EULA=YES timeout 900 $FVENV/bin/python scripts/reinforcement_learning/train.py \
    --rl_library rsl_rl --task=$FULL_SMOKE_TASK --num_envs=256 --max_iterations=2 physics=physx --headless \
    > /root/smoke_full_physx.log 2>&1; echo \$?")
  [ "$rc" = 0 ] || { rrun "tail -30 /root/smoke_full_physx.log"; die "physx smoke rc=$rc"; }
  rrun "grep -aE 'Steps per second|Training time' /root/smoke_full_physx.log | tail -3"
elif [ "$SKIP_SMOKE" = 0 ]; then
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
if [ "$FULL" = 1 ]; then
  rrun "$FVENV/bin/pip list 2>/dev/null | grep -iE '^(isaacsim|isaaclab|newton|torch) ' || true"
  cat <<SUMMARY

FULL Isaac Lab is ready on $DEST:

  ssh -p $PORT $DEST
  source $FVENV/bin/activate && cd $FISAACLAB
  python scripts/reinforcement_learning/train.py --rl_library rsl_rl \\
      --task=Isaac-Ant-Direct-v0 --num_envs=4096 --headless physics=physx

Notes:
  - always --headless (no display); physics=physx = full Isaac Sim,
    physics=newton_mjwarp = kit-less fast path
  - EULA already accepted via marker; OMNI_KIT_ACCEPT_EULA=YES is the fallback
  - the data disk holds ~35-40 GB of the 50 GB -- watch checkpoints in
    $FISAACLAB/logs/ and purge when tight (nothing on the data disk survives
    an instance release, unlike /root in a saved image)
SUMMARY
else
  rrun "$RVENV/bin/pip list 2>/dev/null | grep -iE '^(isaaclab|newton|mujoco-warp|warp-lang|torch|rsl-rl-lib|usd-core|LEAP)' || true"
  cat <<SUMMARY

Kit-less Isaac Lab is ready on $DEST. To train there:

  ssh -p $PORT $DEST
  cd $RREPO
  /root/isaaclab-kitless/bin/python scripts/rsl_rl/train_kitless.py \\
      --task $SMOKE_TASK --headless presets=$SMOKE_PRESET

Only tasks with a kit-less physics preset (e.g. presets=$SMOKE_PRESET) can run on
this box; PhysX/Kit-rendered tasks (the LEAP repose/direct set) need Isaac Sim --
re-run this script with --full to provision that stack side-by-side.
SUMMARY
fi
