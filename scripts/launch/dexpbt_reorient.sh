#!/bin/bash
# One process per member. No implicit session termination or remote deployment.
set -euo pipefail
POP="${POP:-4}"
ENVS="${ENVS:-512}"
EPOCHS="${EPOCHS:-50000}"
INTERVAL="${INTERVAL:-1000000}"
BASE="${BASE:-$HOME/dexpbt_reorient}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/env_isaaclab/bin/python}"
RESUME="${RESUME:-0}"
TASK="${TASK:-Isaac-Dexsuite-Kuka-Allegro-Reorient-PBT-v0}"
DRY_RUN="${DRY_RUN:-0}"
[[ "$POP" =~ ^[1-9][0-9]*$ ]] || { printf 'POP must be positive\n' >&2; exit 2; }
mkdir -p "$BASE"
BASE="$(cd "$BASE" && pwd)"
workspace="$BASE/logs/rl_games/dexpbt_reorient/pbt_workspace"
if [[ "$RESUME" != 1 && -d "$workspace" ]] && [[ -n "$(find "$workspace" -name '*.yaml' -print -quit)" ]]; then
    printf 'Refusing a fresh population in a populated workspace; use RESUME=1 or a new BASE\n' >&2
    exit 2
fi
if [[ "$DRY_RUN" != 1 ]]; then
    for ((i=0; i<POP; i++)); do
        if tmux has-session -t "dexpbt-$i" 2>/dev/null; then
            printf 'Refusing to replace existing tmux session dexpbt-%s\n' "$i" >&2
            exit 2
        fi
    done
fi
commands=()
for ((i=0; i<POP; i++)); do
    cmd=("$PYTHON" "$REPO/scripts/rl_games/train_dexpbt_member.py"
        --task "$TASK" --num_envs "$ENVS" --max_iterations "$EPOCHS" --seed "$((42+i))" --headless
        "pbt.policy_idx=$i" "pbt.num_policies=$POP" "pbt.directory=$BASE" "pbt.interval_steps=$INTERVAL")
    if [[ "$RESUME" == 1 ]]; then
        printf -v slot '%03d' "$i"
        checkpoint="$("$PYTHON" - "$BASE/logs/rl_games/dexpbt_reorient/pbt_workspace/$slot" <<'PY'
import pathlib, sys, yaml
root = pathlib.Path(sys.argv[1])
candidates = []
for path in root.glob('*.yaml'):
    try:
        data = yaml.safe_load(path.read_text())
        checkpoint = pathlib.Path(data['checkpoint'])
        if path.stem.isdigit() and checkpoint.is_file():
            candidates.append((int(data['frame']), 0, str(checkpoint)))
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
        continue
# A staged exploit may be newer than the last successfully restarted publication.
path = root / 'restart/checkpoint.yaml'
if path.is_file():
    data = yaml.safe_load(path.read_text())
    if pathlib.Path(data['checkpoint']).is_file():
        candidates.append((int(data.get('recipient_frame', data['frame'])), 1, data['checkpoint']))
if candidates:
    print(max(candidates)[2])
PY
)"
        [[ -n "$checkpoint" ]] || { printf 'No committed checkpoint for member %s\n' "$i" >&2; exit 2; }
        cmd+=(--checkpoint "$checkpoint" --pbt-resume "${checkpoint%.pth}.yaml")
    fi
    # Forward user overrides as intact argv tokens (paths with spaces are supported).
    cmd+=("$@")
    printf -v quoted '%q ' "${cmd[@]}"
    printf -v repo_q '%q' "$REPO"
    printf -v log_q '%q' "$BASE/member_$i.log"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '%s\n' "$quoted"
        continue
    fi
    commands+=("cd $repo_q && exec $quoted >> $log_q 2>&1")
done
# Prepare EVERY member before launching any. Automatic retry of an old command
# could silently undo an exploit; failures instead require explicit RESUME=1.
for ((i=0; i<${#commands[@]}; i++)); do
    tmux new-session -d -s "dexpbt-$i" "${commands[$i]}"
    printf 'Launched member %s/%s in tmux dexpbt-%s\n' "$i" "$POP" "$i"
    sleep 15
done
