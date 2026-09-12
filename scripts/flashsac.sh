#!/usr/bin/env bash
# Unified FlashSAC launcher (TUI): pick task, train or play, PhysX or Newton.
#
# Replaces the four previous wrappers (flashsac_{train,play}_{leap_direct,dexsuite_lift}.sh).
# With no arguments it is a menu-driven TUI; every choice can also be given on
# the command line for scripted use (mode positional first, then flags).
# Anything the script does not recognize is forwarded
# verbatim (train: appended to train.py like `--overrides ...`; play: forwarded
# to play_isaaclab.py, e.g. `--overrides seed=1000`).
#
# Tasks are FULL gym task ids. The known ids below get tuned defaults (group /
# exp / play-env / train profile); ANY other gym id also works with generic
# defaults, so future environments need no script change:
#   Isaac-Repose-Cube-LEAP-Direct-v0            (LEAP in-hand cube)
#   Isaac-Repose-Objects-LEAP-Direct-v0         (multi-object catalog)
#   Isaac-Repose-Objects-LEAP-Direct-v1         (objects + manifest inertials)
#   Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0  (KUKA+Allegro pick-lift)
#
# Backends (training only; play always opens the Isaac Sim GUI = PhysX):
#   physx    Isaac Sim + PhysX (headless) -- the default, matches all existing
#            checkpoints and the measured benchmark numbers.
#   newton   Newton (newton_mjwarp preset), NO Isaac Sim process: seconds to
#            launch, ~8-10 GB RAM freed (bigger pinned replay buffers possible).
#            STATUS 2026-09-11 (see AGENTS.md "kit-less"): cube/objects/v1 build,
#            step and hold the object on Newton, and v1's manifest inertial writes
#            work -- but the PULP CONTACT SENSORS READ ZERO in both contact modes
#            (isaaclab_newton beta reporting gap), so the 12 tactile obs channels
#            and reward terms 8/9/10 are dead. lift has no Newton support at all.
#
# Usage:
#   scripts/flashsac.sh                                          # interactive menus
#   scripts/flashsac.sh train --task Isaac-Repose-Objects-LEAP-Direct-v0 --backend newton
#   scripts/flashsac.sh play --task Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0 -n 4
#   scripts/flashsac.sh train --task <any-gym-task-id> --dry-run   # print the exec line only
#
# Options:
#   train | play           the MODE, as a (first) positional argument
#   -t, --task ID          full gym task id (known ids listed in the menu plus a
#                          free-text "other" entry for anything else)
#   -b, --backend BACKEND  physx | newton          (train only)
#   --exp NAME             experiment name (train: TB/checkpoint dir; play: filter)
#   --group NAME           model group (default depends on task)
#   -c, --checkpoint DIR   play: explicit checkpoint dir (skips the run picker)
#   -n, --num-envs N       play: parallel envs (default 8)
#   --num-episodes N       play: episodes before exit (default 50)
#   --dry-run              print the command instead of exec'ing it
#   -h, --help             this help
#
# Everything else stays env-var tunable like the old wrappers (values already
# exported in your shell win over the built-in defaults): FLASHSAC_DIR, PYTHON,
# NUM_ENV_STEPS, NUM_TRAIN_ENVS, BUFFER_LEN, BUFFER_DEVICE, UPDATES_PER_STEP,
# SAVE_EVERY, LOG_EVERY, METRICS_EVERY, HOLD_TIME, ACT_SMOOTH, GROUP, EXP,
# NUM_ENVS, NUM_EPISODES, CHECKPOINT. Note a stale exported FLASH_RL_PRESET is
# deliberately UNSET in physx mode so it cannot silently switch the backend.

set -euo pipefail

FLASHSAC_DIR="${FLASHSAC_DIR:-$HOME/Documents/projects/FlashSAC}"
PYTHON="${PYTHON:-$HOME/env_isaaclab/bin/python}"

# ---------------------------------------------------------------- task table --
# Tuned defaults for the tasks we train. Keys are FULL gym task ids; anything
# else falls through to the generic defaults (group "flashsac", exp "run1",
# play env = train env, laptop-ish 1024-env/2M-cpu profile, newton allowed
# but flagged UNVERIFIED -- the task cfg must ship a newton_mjwarp preset).
CUBE_TASK="Isaac-Repose-Cube-LEAP-Direct-v0"
OBJECTS_TASK="Isaac-Repose-Objects-LEAP-Direct-v0"
OBJECTS_V1_TASK="Isaac-Repose-Objects-LEAP-Direct-v1"
LIFT_TASK="Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-v0"
declare -A PLAY_TASK_ID GROUP_DEF EXP_DEF NEWTON_OK TASK_DESC
PLAY_TASK_ID[$CUBE_TASK]="$CUBE_TASK"           # leap tasks: same id at play
PLAY_TASK_ID[$OBJECTS_TASK]="$OBJECTS_TASK"
PLAY_TASK_ID[$OBJECTS_V1_TASK]="$OBJECTS_V1_TASK"
PLAY_TASK_ID[$LIFT_TASK]="Isaac-Dexsuite-Kuka-Allegro-Lift-Direct-Play-v0"  # ADR max difficulty
GROUP_DEF[$CUBE_TASK]=leap_direct
GROUP_DEF[$OBJECTS_TASK]=leap_objects
GROUP_DEF[$OBJECTS_V1_TASK]=leap_objects
GROUP_DEF[$LIFT_TASK]=dexsuite
EXP_DEF[$CUBE_TASK]=run1
EXP_DEF[$OBJECTS_TASK]=objects-r1
EXP_DEF[$OBJECTS_V1_TASK]=objects-v1
EXP_DEF[$LIFT_TASK]=lift-r1
NEWTON_OK[$CUBE_TASK]=1
NEWTON_OK[$OBJECTS_TASK]=1
NEWTON_OK[$OBJECTS_V1_TASK]=1
NEWTON_OK[$LIFT_TASK]=0                        # PhysX-only contact sensors by design
TASK_DESC[$CUBE_TASK]="LEAP in-hand cube reorientation"
TASK_DESC[$OBJECTS_TASK]="multi-object catalog reorientation"
TASK_DESC[$OBJECTS_V1_TASK]="objects + manifest inertials (m,COM,I)"
TASK_DESC[$LIFT_TASK]="KUKA+Allegro pick-from-table-and-lift"

# ------------------------------------------------------------------- helpers --
usage() { awk 'NR>1 && !/^#/ {exit} NR>1 {sub(/^# ?/, ""); print}' "$0"; }

prompt_choice() { # prompt_choice VAR "question" opt1 opt2 ...
    local __var=$1 __q=$2; shift 2
    local -a __opts=("$@") __i
    while true; do
        read -rp "$__q [$*]: " __i || exit 130
        __i="${__i:-${__opts[0]}}"
        local __o
        for __o in "${__opts[@]}"; do
            [ "$__i" = "$__o" ] && { printf -v "$__var" '%s' "$__i"; return 0; }
        done
        # also accept the 1-based option number ("1" == first option)
        case "$__i" in
            (*[!0-9]*|'') : ;;
            (*) [ "$__i" -ge 1 ] && [ "$__i" -le ${#__opts[@]} ] \
                && { printf -v "$__var" '%s' "${__opts[$((__i - 1))]}"; return 0; } ;;
        esac
        echo "  please pick one of: $*" >&2
    done
}

prompt_value() { # prompt_value VAR "question" default
    local __var=$1 __q=$2 __d=$3 __v
    read -rp "$__q [${__d}]: " __v || exit 130
    printf -v "$__var" '%s' "${__v:-$__d}"
}

# ------------------------------------------------------------------ arg parse --
TASK="" MODE="" BACKEND="" CHECKPOINT="${CHECKPOINT:-}" DRY_RUN=0 FORWARD=()
while [ $# -gt 0 ]; do
    case "$1" in
        train|play) MODE="$1"; shift ;;
        -m|--mode|--mode=*)
            echo "error: --mode is gone; pass the mode as a positional: '$0 train ...' / '$0 play ...'" >&2
            exit 1 ;;
        -t|--task) TASK="$2"; shift 2 ;;
        -b|--backend) BACKEND="$2"; shift 2 ;;
        --task=*) TASK="${1#--task=}"; shift ;;
        --backend=*) BACKEND="${1#--backend=}"; shift ;;
        --exp) EXP="$2"; shift 2 ;;
        --exp=*) EXP="${1#--exp=}"; shift ;;
        --group) GROUP="$2"; shift 2 ;;
        --group=*) GROUP="${1#--group=}"; shift ;;
        -c|--checkpoint) CHECKPOINT="$2"; shift 2 ;;
        -n|--num-envs) NUM_ENVS="$2"; shift 2 ;;
        --num-envs=*) NUM_ENVS="${1#--num-envs=}"; shift ;;
        --num-episodes) NUM_EPISODES="$2"; shift 2 ;;
        --num-episodes=*) NUM_EPISODES="${1#--num-episodes=}"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) FORWARD+=("$1"); shift ;;
    esac
done

# --------------------------------------------------------------- the menus ----
if [ -z "$TASK" ]; then
    echo
    echo "Task (full gym id):"
    i=1
    for t in "$CUBE_TASK" "$OBJECTS_TASK" "$OBJECTS_V1_TASK" "$LIFT_TASK"; do
        kl="newton NO "
        [ "${NEWTON_OK[$t]}" = 1 ] && kl="newton ok"
        echo "  $i) $t -- ${TASK_DESC[$t]:-} ($kl)"
        i=$((i + 1))
    done
    echo "  5) other -- type any gym task id (generic defaults)"
    while true; do
        read -rp "Pick task [1-5]: " pick || exit 130
        case "$pick" in
            1) TASK="$CUBE_TASK"; break ;;
            2) TASK="$OBJECTS_TASK"; break ;;
            3) TASK="$OBJECTS_V1_TASK"; break ;;
            4) TASK="$LIFT_TASK"; break ;;
            5) while true; do
                   read -rp "  gym task id: " TASK || exit 130
                   [ -n "$TASK" ] && break
                   echo "  empty id, try again" >&2
               done
               break ;;
            *) echo "  1-5 please" >&2 ;;
        esac
    done
fi
[ -n "$TASK" ] || { echo "error: empty task id" >&2; exit 1; }
# friendly migration: the pre-full-id nicknames would otherwise train a bogus id
case "$TASK" in
    cube|objects|objects-v1|lift)
        echo "error: '$TASK' is a nickname; use the full gym task id:" >&2
        echo "  cube        -> $CUBE_TASK" >&2
        echo "  objects     -> $OBJECTS_TASK" >&2
        echo "  objects-v1  -> $OBJECTS_V1_TASK" >&2
        echo "  lift        -> $LIFT_TASK" >&2
        exit 1 ;;
esac

if [ -z "$MODE" ]; then
    prompt_choice MODE "Train or play?" train play
fi
case "$MODE" in
    train|play) ;;
    *) echo "error: --mode must be train or play" >&2; exit 1 ;;
esac

if [ "$MODE" = train ]; then
    if [ -z "$BACKEND" ]; then
        case "${NEWTON_OK[$TASK]:-}" in
            1) prompt_choice BACKEND "Physics backend? (physx = Isaac Sim+PhysX, newton = Newton)" physx newton ;;
            0) BACKEND=physx
               echo "(task '$TASK' has no Newton support -- PhysX only)" ;;
            *) prompt_choice BACKEND "Physics backend? (newton on this task is UNVERIFIED)" physx newton ;;
        esac
    fi
    case "$BACKEND" in
        physx|newton) ;;
        kit|kitless)
            echo "error: backend values are 'physx' and 'newton' now (formerly kit/kitless)" >&2
            exit 1 ;;
        *) echo "error: --backend must be physx or newton" >&2; exit 1 ;;
    esac
    if [ "$BACKEND" = newton ]; then
        if [ "${NEWTON_OK[$TASK]:-}" = 0 ]; then
            echo "error: task '$TASK' has no Newton support (PhysX-only contact sensors)" >&2
            exit 1
        fi
        if [ -z "${NEWTON_OK[$TASK]:-}" ]; then
            echo "  NOTE newton on '$TASK' is UNVERIFIED: the task cfg must ship a"
            echo "       newton_mjwarp preset (official Isaac Lab tasks do; ours: cube/objects/v1)."
        fi
    fi
else
    if [ -n "$BACKEND" ] && [ "$BACKEND" = newton ]; then
        echo "error: play always opens the Isaac Sim GUI (PhysX); newton play is not a thing" >&2
        exit 1
    fi
    BACKEND=physx
fi

# ----------------------------------------------------- per-task knob defaults --
# every lookup falls back for tasks outside the table (generic profile)
GROUP="${GROUP:-${GROUP_DEF[$TASK]:-flashsac}}"
# train: default the experiment name per task; play: no default (empty = every
# exp under the group -- the run picker shows the exp name anyway)
if [ "$MODE" = train ]; then
    EXP="${EXP:-${EXP_DEF[$TASK]:-run1}}"
else
    EXP="${EXP:-}"
fi

if [ "$MODE" = train ]; then
    # LEAP profile: 4096 envs / 6M CUDA buffer (12 GB+ GPU, e.g. the 4090 remote);
    # for the 12 GB laptop use the env vars (512 envs / 1M, see AGENTS.md)
    case "$TASK" in
        "$CUBE_TASK"|"$OBJECTS_TASK"|"$OBJECTS_V1_TASK")
            DEF_STEPS=850485760; DEF_ENVS=4096; DEF_BUFFER=6000000; DEF_BUFDEV=cuda
            DEF_UPDATES=1; DEF_LOG_EVERY=""; DEF_METRICS_EVERY="" ;;
        "$LIFT_TASK")
            # laptop profile: pinned-RAM buffer (2M ~12.8 GB host RAM is the ceiling)
            DEF_STEPS=1004857600; DEF_ENVS=1024; DEF_BUFFER=2000000; DEF_BUFDEV=cpu
            DEF_UPDATES=1 ;;
        *)
            # generic: laptop-safe profile with pinned TB cadences
            DEF_STEPS=850485760; DEF_ENVS=1024; DEF_BUFFER=2000000; DEF_BUFDEV=cpu
            DEF_UPDATES=1 ;;
    esac
    NUM_ENV_STEPS="${NUM_ENV_STEPS:-$DEF_STEPS}"
    NUM_TRAIN_ENVS="${NUM_TRAIN_ENVS:-$DEF_ENVS}"
    BUFFER_LEN="${BUFFER_LEN:-$DEF_BUFFER}"
    BUFFER_DEVICE="${BUFFER_DEVICE:-$DEF_BUFDEV}"
    UPDATES_PER_STEP="${UPDATES_PER_STEP:-$DEF_UPDATES}"
    # SAVE/LOG/METRICS count CONTROL ITERATIONS (one iteration = all envs stepped
    # once); auto-scale to ~10.5M env-steps per checkpoint / ~1M per log point /
    # ~0.5M per metrics point (lift pins logging cadences; leap tasks keep the
    # upstream defaults unless overridden)
    SAVE_EVERY="${SAVE_EVERY:-$((10485760 / NUM_TRAIN_ENVS))}"
    case "$TASK" in
        "$CUBE_TASK"|"$OBJECTS_TASK"|"$OBJECTS_V1_TASK")
            LOG_EVERY="${LOG_EVERY:-}"
            METRICS_EVERY="${METRICS_EVERY:-}" ;;
        *)
            LOG_EVERY="${LOG_EVERY:-$((1048576 / NUM_TRAIN_ENVS))}"
            METRICS_EVERY="${METRICS_EVERY:-$((524288 / NUM_TRAIN_ENVS))}" ;;
    esac
    case "$TASK" in
        "$CUBE_TASK"|"$OBJECTS_TASK"|"$OBJECTS_V1_TASK") DEF_HOLD="0.2:1.5"; DEF_SMOOTH=0.2 ;;
        *) DEF_HOLD=0; DEF_SMOOTH=1.0 ;;   # off by default (hold hook is leap-only anyway)
    esac
    HOLD_TIME="${HOLD_TIME:-$DEF_HOLD}"
    ACT_SMOOTH="${ACT_SMOOTH:-$DEF_SMOOTH}"

    echo
    echo "== FlashSAC train | task=$TASK | backend=$BACKEND =="
    prompt_value GROUP "  model group" "$GROUP"
    prompt_value EXP "  experiment name" "$EXP"
    prompt_value NUM_TRAIN_ENVS "  parallel envs" "$NUM_TRAIN_ENVS"
    prompt_value NUM_ENV_STEPS "  total env steps" "$NUM_ENV_STEPS"
    prompt_value BUFFER_LEN "  replay buffer (transitions)" "$BUFFER_LEN"
    prompt_value BUFFER_DEVICE "  buffer device (cuda|cpu)" "$BUFFER_DEVICE"
    if [ "$BACKEND" = newton ] && [ "${NEWTON_OK[$TASK]:-}" = 1 ]; then
        echo "  NOTE newton backend: pulp tactile channels read 0 (isaaclab_newton"
        echo "       reporting gap) -- obs terms 12 tactile dims + reward terms 8/9/10 dead."
    fi
    read -rp "  Launch? [Y/n] " go || exit 130
    if [ "$go" = n ] || [ "$go" = N ]; then exit 0; fi
else
    NUM_ENVS="${NUM_ENVS:-8}"
    NUM_EPISODES="${NUM_EPISODES:-50}"
    HOLD_TIME="${HOLD_TIME:-1.0}"   # play_isaaclab defaults; hold is a no-op on lift
    ACT_SMOOTH="${ACT_SMOOTH:-0.3}"
    EXP="${EXP:-}"                  # play: empty = every exp under the group

    PLAY_ENV="${PLAY_TASK_ID[$TASK]:-$TASK}"
    if [ -z "${PLAY_TASK_ID[$TASK]:-}" ]; then
        # unknown task: give the -Play- variant a chance (lift-style eval envs)
        prompt_value PLAY_ENV "  play env id (e.g. a -Play- variant)" "$TASK"
    fi

    echo
    echo "== FlashSAC play | task=$TASK ($PLAY_ENV) | Isaac Sim GUI =="
    prompt_value GROUP "  model group" "$GROUP"
    prompt_value EXP "  experiment filter (empty = any)" "$EXP"

    MODELS_ROOT="$FLASHSAC_DIR/models/$GROUP"
    TRAIN_TASK="$TASK"
    if [ -z "$CHECKPOINT" ]; then
        # expand the glob FIRST: a bare `ls -td` with no args lists the CURRENT
        # DIRECTORY, which would fabricate a bogus "." run
        shopt -s nullglob
        run_dirs=( "$MODELS_ROOT"/*/"$TRAIN_TASK"/*/ )
        if [ ${#run_dirs[@]} -eq 0 ]; then
            echo "error: no runs found for GROUP=$GROUP TASK=$TRAIN_TASK under $MODELS_ROOT" >&2
            echo "hint: train first (and let it exit), or pass --checkpoint DIR" >&2
            exit 1
        fi
        runs=()
        while IFS= read -r d; do runs+=("$d"); done < <(ls -td "${run_dirs[@]}" 2>/dev/null || true)
        # apply the exp filter after listing so "empty = any" stays natural
        if [ -n "$EXP" ]; then
            filtered=()
            for r in "${runs[@]}"; do
                case "$r" in "$MODELS_ROOT/$EXP/"*) filtered+=("$r") ;; esac
            done
            runs=("${filtered[@]}")
        fi
        if [ ${#runs[@]} -eq 0 ]; then
            echo "error: no runs found for GROUP=$GROUP TASK=$TRAIN_TASK under $MODELS_ROOT" >&2
            echo "hint: train first (and let it exit), or pass --checkpoint DIR" >&2
            exit 1
        fi
        echo "  Runs (newest first):"
        i=1
        for r in "${runs[@]}"; do
            echo "    $i) $(basename "$(dirname "$(dirname "$r")")")/$(basename "$r")"
            [ $i -ge 10 ] && break
            i=$((i + 1))
        done
        while true; do
            read -rp "  Pick run [1]: " pick || exit 130
            pick="${pick:-1}"
            case "$pick" in (*[!0-9]*|'') echo "  number please" >&2; continue ;; esac
            [ "$pick" -ge 1 ] && [ "$pick" -le ${#runs[@]} ] && break
            echo "  1-${#runs[@]} please" >&2
        done
        RUN_DIR="${runs[$((pick - 1))]}"
        CHECKPOINT=$(ls -td "${RUN_DIR%/}"/step*/ 2>/dev/null | head -1 || true)
        if [ ! -d "$CHECKPOINT" ]; then
            echo "error: no step*/ checkpoint inside $RUN_DIR" >&2
            exit 1
        fi
    fi
    prompt_value NUM_ENVS "  parallel envs" "$NUM_ENVS"
    prompt_value NUM_EPISODES "  episodes" "$NUM_EPISODES"
    echo "  Playing checkpoint: $CHECKPOINT"
    read -rp "  Launch? [Y/n] " go || exit 130
    if [ "$go" = n ] || [ "$go" = N ]; then exit 0; fi
fi

# ------------------------------------------------------------------- launch ---
cd "$FLASHSAC_DIR"
export FLASH_RL_EXTRA_TASK_MODULES="${FLASH_RL_EXTRA_TASK_MODULES:-LEAP_Isaaclab.tasks}"
export FLASH_RL_HOLD_TIME="$HOLD_TIME"
export FLASH_RL_ACT_MOVING_AVERAGE="$ACT_SMOOTH"
# a stale FLASH_RL_PRESET from an earlier kit-less launch must not silently
# switch the backend of a kit run
if [ "$BACKEND" = newton ]; then
    export FLASH_RL_PRESET=newton_mjwarp
else
    unset FLASH_RL_PRESET
fi
# hybrid-graphics laptop: render / compute on the NVIDIA dGPU.
# Guarded by the ICD file: on headless servers (AutoDL etc.) the path does not
# exist and a bogus VK_ICD_FILENAMES natively crashes Kit at startup.
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    export __NV_PRIME_RENDER_OFFLOAD=1
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
    export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
fi

if [ "$MODE" = train ]; then
    CMD=( "$PYTHON" train.py
        --config_name flashSAC_base
        --overrides env=isaaclab
        --overrides "env.env_name=$TASK"
        --overrides "num_env_steps=$NUM_ENV_STEPS"
        --overrides "num_train_envs=$NUM_TRAIN_ENVS"
        --overrides num_eval_envs=null
        --overrides num_record_envs=null
        --overrides "num_eval_episodes=$NUM_TRAIN_ENVS"
        --overrides num_record_episodes=0
        --overrides recording_per_interaction_step=null
        --overrides "agent.buffer_max_length=$BUFFER_LEN"
        --overrides "agent.buffer_device_type=$BUFFER_DEVICE"
        --overrides "save_checkpoint_per_interaction_step=$SAVE_EVERY"
        --overrides "updates_per_interaction_step=$UPDATES_PER_STEP"
        --overrides gamma=0.99
        --overrides n_step=3
        --overrides "group_name=$GROUP"
        --overrides "exp_name=$EXP"
        --overrides logger_type=tensorboard
    )
    [ -n "${LOG_EVERY:-}" ] && CMD+=( --overrides "logging_per_interaction_step=$LOG_EVERY" )
    [ -n "${METRICS_EVERY:-}" ] && CMD+=( --overrides "metrics_per_interaction_step=$METRICS_EVERY" )
    echo "[flashsac] task=$TASK envs=$NUM_TRAIN_ENVS buffer=$BUFFER_LEN@$BUFFER_DEVICE " \
         "steps=$NUM_ENV_STEPS save_every_iters=$SAVE_EVERY group=$GROUP exp=$EXP backend=$BACKEND"
else
    CMD=( "$PYTHON" play_isaaclab.py
        --checkpoint_path "$CHECKPOINT"
        --num_envs "$NUM_ENVS"
        --num_episodes "$NUM_EPISODES"
        --hold_time "$HOLD_TIME"
        --action_smooth "$ACT_SMOOTH"
        --overrides env=isaaclab
        --overrides "env.env_name=$PLAY_ENV"
        --overrides gamma=0.99
        --overrides n_step=3
        --overrides agent.use_compile=false
        --overrides agent.use_amp=false
        --overrides agent.buffer_max_length=10000
        --overrides agent.buffer_min_length=10
        --overrides agent.buffer_device_type=cpu
    )
    echo "[flashsac] play task=$PLAY_ENV ($NUM_ENVS envs, $NUM_EPISODES episodes) " \
         "ckpt=$CHECKPOINT hold=$HOLD_TIME smooth=$ACT_SMOOTH"
fi
CMD+=( ${FORWARD[@]+"${FORWARD[@]}"} )

if [ "$DRY_RUN" = 1 ]; then
    printf '[dry-run]'; printf ' %q' "${CMD[@]}"; echo
else
    exec "${CMD[@]}"
fi
