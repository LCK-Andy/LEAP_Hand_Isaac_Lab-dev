"""Stage B tracking-first completed-episode selection; never consumes Stage A extras.

Reuse Stage A's validated sufficient-statistics implementation internally, not
its environment/task identity or public metric names. Any overloaded episode
gets zero tracking and retention credit. Donors may not increase overload rate
or reduce support-free retention; significant tracking improvement wins first,
with retention used only for exact tracking ties.
"""

from .grasp import GraspHoldWindow, select_grasp_donor


def validate_motion_environment(task, env):
    """Require the Stage B measurement/state contract without changing the MDP."""
    if task != "Isaac-Grasp-Motion-Cube-LEAP-Direct-v0":
        raise ValueError("grasp_motion requires the native Stage B training task")
    if getattr(env, "_physics_backend", None) not in {"physx", "newton"}:
        raise ValueError("grasp_motion backend must support the full measurement contract")
    if getattr(env, "grasp_motion_contract_version", None) != 1:
        raise ValueError("grasp_motion requires environment measurement contract version 1")
    if not all(callable(getattr(env, name, None)) for name in ("get_env_state", "set_env_state")):
        raise ValueError("grasp_motion requires explicit environment checkpoint state hooks")


class GraspMotionWindow:
    """Motion-only schema adapter over the common safe episode statistics."""

    def __init__(self):
        self._episodes = GraspHoldWindow()

    def consume(self):
        self._episodes.consume()

    def add(self, infos, frame, now=None):
        stats = infos.get("grasp_motion")
        if not stats:
            return
        if not isinstance(stats, dict):
            raise ValueError("grasp_motion requires a metric dictionary")
        adapted = {
            "condition_key": stats.get("condition_key"),
            "completed_valid_fraction": stats.get("completed_tracking_fraction"),
            "completed_retained": stats.get("completed_retained"),
            "completed_overload_fraction": stats.get("completed_overload_fraction"),
        }
        try:
            self._episodes.add({"grasp_hold": adapted}, frame, now)
        except ValueError as error:
            raise ValueError(str(error).replace("grasp_hold", "grasp_motion")) from error

    def ready(self, cfg, frame, now=None):
        stats = self._episodes.ready(cfg, frame, now).get("grasp_hold")
        if not stats:
            return {}
        return {
            "grasp_motion": {
                **stats,
                "success_rate": stats["progress"],
                "variance": stats["progress_variance"],
                "progress": stats["success_rate"],
                "progress_variance": stats["variance"],
            }
        }


def select_motion_donor(records, own, cfg, now=None):
    """Tracking-first comparison, with an additional no-worse-retention guard."""
    ours = own.get("evaluations", {}).get("grasp_motion")
    if not ours:
        return None

    def adapted(record, stats):
        return {**record, "evaluations": {"grasp_hold": stats}}

    candidates = {}
    for policy, record in records.items():
        stats = (record or {}).get("evaluations", {}).get("grasp_motion")
        if stats and stats["progress"] >= ours["progress"]:
            candidates[policy] = adapted(record, stats)
    return select_grasp_donor(candidates, adapted(own, ours), cfg, now)
