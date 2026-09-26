"""Completed-episode asynchronous PBT with recipient-owned restart transactions."""

import json
import os
import random
import sys
import time

from rl_games.common.algo_observer import AlgoObserver

from . import pbt_utils
from .mutation import mutate
from .pbt_cfg import PbtCfg
from .runtime import (
    PREFIX,
    SUPPORTED,
    apply_overrides,
    budget_reached,
    effective_params,
    install_restore,
    install_train_guard,
    restart_command,
    resume_metadata,
)


def initialize_population(params, resumed=False):
    """Diversify new members before runner/observer construction; member zero is the baseline."""
    cfg = PbtCfg(**params["pbt"])
    values = pbt_utils.filter_params(pbt_utils.flatten_dict({"agent": params}), cfg.mutation)
    if cfg.enabled and cfg.initial_diversity and cfg.policy_idx != 0 and not resumed:
        values = mutate(
            values,
            cfg.mutation,
            1.0,
            (1.1, 1.5) if cfg.objective == "allegro_goals" else cfg.change_range,
            cfg.mutation_bounds,
            rng=random.Random(int(params["params"]["seed"]) + cfg.policy_idx),
        )
        apply_overrides(params, values)
        print(f"[pbt] Initial diversity member {cfg.policy_idx}: {values}", flush=True)
    return values


class PbtAlgoObserver(AlgoObserver):
    """Saves complete PPO state and only exploits fresh comparable completed episodes."""

    def __init__(self, params, args_cli):
        super().__init__()
        if getattr(args_cli, "distributed", False) or int(os.environ.get("WORLD_SIZE", 1)) != 1:
            raise ValueError("DexPBT supports one process/GPU per member, not distributed PPO")
        self.cfg = PbtCfg(**params["pbt"])
        self.original_argv = list(args_cli.original_argv)
        self.resume = getattr(args_cli, "pbt_resume_metadata", None)
        self.plain_checkpoint = bool(getattr(args_cli, "checkpoint", None)) and self.resume is None
        self.params = params
        self.rng = pbt_utils.recipient_rng(params["params"]["seed"], self.cfg.policy_idx, self.resume)
        config = params["params"]["config"]
        self.pbt_params = {PREFIX + key: float(config[key]) for key in SUPPORTED if key in config}
        if self.cfg.enabled and not self.cfg.mutation:
            raise ValueError("PBT mutation parameter dictionary is empty")
        if not set(self.cfg.mutation).issubset(self.pbt_params):
            raise ValueError("Mutation keys must name supported live PPO config parameters")
        from .allegro import GoalWindow
        from .contact import ContactWindow
        from .grasp import GraspHoldWindow
        from .grasp_motion import GraspMotionWindow

        self.count_objective = self.cfg.objective in {"allegro_goals", "contact_goals", "grasp_hold", "grasp_motion"}
        self.window = (
            {
                "allegro_goals": GoalWindow,
                "contact_goals": ContactWindow,
                "grasp_hold": GraspHoldWindow,
                "grasp_motion": GraspMotionWindow,
            }[self.cfg.objective]()
            if self.count_objective
            else pbt_utils.EpisodeWindow(self.cfg)
        )
        self.pbt_it = None
        self._rollout_start_frame = None
        self._rollout_completed_frames = 0
        self.last_exploit_frame = (self.resume or {}).get("last_exploit_frame", 0)
        self.evidence = dict((self.resume or {}).get("selection_evidence", {}))
        self.lineage = (self.resume or {}).get("lineage", f"member-{self.cfg.policy_idx}")
        self.segment_rollouts = 0
        self.process_started_at = getattr(args_cli, "process_started_at", time.time())
        self.segment_started_at = time.monotonic()
        self.replacement_reported = False

    def after_init(self, algo):
        self.algo = algo
        self.ws_dir = os.path.join(algo.train_dir, self.cfg.workspace)
        self.curr_policy_dir = os.path.join(self.ws_dir, f"{self.cfg.policy_idx:03d}")
        if self.cfg.enabled:
            os.makedirs(self.curr_policy_dir, exist_ok=True)
        install_restore(algo, self.pbt_params, self.resume)
        restore_with_params = algo.restore

        def restore_without_evidence(checkpoint, *args, **kwargs):
            self.window.consume()
            self._rollout_start_frame = None
            self._rollout_completed_frames = 0
            return restore_with_params(checkpoint, *args, **kwargs)

        algo.restore = restore_without_evidence
        if hasattr(algo, "train"):
            install_train_guard(algo, self.after_print_stats)
        if self.plain_checkpoint:
            restore = algo.restore

            def restore_plain_checkpoint(checkpoint, *args, **kwargs):
                result = restore(checkpoint, *args, **kwargs)
                # A raw PPO checkpoint has no companion stream. Its age is only
                # available AFTER restore; do not silently reuse fresh seed draws.
                self.rng = pbt_utils.recipient_rng(
                    self.params["params"]["seed"],
                    self.cfg.policy_idx,
                    {"frame": int(algo.frame), "epoch": int(algo.epoch_num)},
                )
                self.pbt_it = int(algo.frame) // self.cfg.interval_steps
                return result

            algo.restore = restore_plain_checkpoint

    def process_infos(self, infos, done_indices):
        if self.cfg.enabled:
            frame = int(self.algo.frame)
            if not self.count_objective:
                # RL-Games leaves frame at rollout START until PPO finishes. This
                # callback runs after each vector step, even when no episodes end.
                if self._rollout_start_frame != frame:
                    self._rollout_start_frame = frame
                    self._rollout_completed_frames = 0
                self._rollout_completed_frames += int(getattr(self.algo, "num_actors", 0))
                frame += self._rollout_completed_frames
            self.window.add(infos, frame)

    def after_steps(self):
        # rl-games calls after_steps BEFORE PPO updates and before incrementing frame.
        # Publish in after_print_stats instead so weights/optimizer and frame agree.
        self.segment_rollouts += 1
        self._rollout_start_frame = None
        self._rollout_completed_frames = 0

    def _report_selection(self, own, diagnostics):
        current = int(self.algo.frame)
        if own is not None:
            own["selection_diagnostics"] = diagnostics
            pbt_utils.atomic_yaml(own["pbt_checkpoint"], own)
        print(
            "[pbt-selection] " + json.dumps({"member": self.cfg.policy_idx, "frame": current, **diagnostics}),
            flush=True,
        )
        writer = self.algo.writer
        if hasattr(writer, "add_scalar"):
            reason = diagnostics["reason"]
            previous = getattr(self, "_selection_reason", None)
            if previous and previous != reason:
                writer.add_scalar(f"pbt/selection_reason/{previous}", 0, current)
            writer.add_scalar(f"pbt/selection_reason/{reason}", 1, current)
            self._selection_reason = reason
            for name in ("donor_records", "eligible_donors", "comparable_pairs"):
                writer.add_scalar(f"pbt/{name}", diagnostics.get(name, 0), current)
            if own is not None:
                writer.add_scalar("pbt/objective_valid", int(own["objective_valid"]), current)
                writer.add_scalar("pbt/fresh_episodes", own["evaluation_diagnostics"]["fresh_episodes"], current)
                if own["objective_valid"]:
                    writer.add_scalar("pbt/objective", own["true_objective"], current)

    def after_print_stats(self, frame, epoch_num, total_time):
        if not self.cfg.enabled:
            iteration = int(self.algo.frame) // self.cfg.interval_steps
            if iteration != getattr(self, "_disabled_iteration", None):
                self._report_selection(None, {"reason": "pbt_disabled", "donor": None})
                self._disabled_iteration = iteration
            return
        current = int(self.algo.frame)
        if self.segment_rollouts:
            phase = "cold_first_19" if self.segment_rollouts <= 19 else "steady"
            writer = self.algo.writer
            if hasattr(writer, "add_scalar"):
                writer.add_scalar("pbt/cold_first_19", int(self.segment_rollouts <= 19), current)
                writer.add_scalar("pbt/segment_rollout", self.segment_rollouts, current)
            if self.segment_rollouts in (1, 19, 20):
                print(
                    f"[pbt] measurement_phase={phase} rollout={self.segment_rollouts} frame={current} raw_unchanged=true",
                    flush=True,
                )
            if not self.replacement_reported:
                event = {
                    "event": "segment_first_update",
                    "frame": current,
                    "lineage": self.lineage,
                    "process_to_first_update_seconds": time.time() - self.process_started_at,
                    "observer_to_first_update_seconds": time.monotonic() - self.segment_started_at,
                }
                if (self.resume or {}).get("replacement_started_at") is not None:
                    event["replacement_to_first_update_seconds"] = time.time() - self.resume["replacement_started_at"]
                print("[pbt-profile] " + json.dumps(event), flush=True)
                self.replacement_reported = True
        terminal = budget_reached(self.algo)
        iteration = current // self.cfg.interval_steps
        if self.pbt_it is None:
            restored_frame = (self.resume or {}).get("recipient_frame", (self.resume or {}).get("frame", 0))
            self.pbt_it = int(restored_frame) // self.cfg.interval_steps
        if iteration <= self.pbt_it and not terminal:
            return
        if terminal and os.path.exists(os.path.join(self.curr_policy_dir, "final.yaml")):
            saved = resume_metadata(os.path.join(self.curr_policy_dir, "final.pth"))
            if saved and saved["frame"] == current and saved["epoch"] == self.algo.epoch_num:
                return
        self.pbt_it = iteration
        evaluations = self.window.ready(self.cfg, current)
        # Publish even during warm-up/no resets, enabling exact manual resume, but
        # such records carry no selectable score until the complete window is ready.
        self.pbt_params = effective_params(self.algo, self.pbt_params)
        score = max(
            (g["success_rate"] + self.cfg.progress_weight * g["progress"] for g in evaluations.values()), default=0.0
        )
        own = pbt_utils.save_pbt_checkpoint(
            self.curr_policy_dir,
            score,
            iteration,
            self.algo,
            self.pbt_params,
            policy_idx=self.cfg.policy_idx,
            evaluations=evaluations,
            objective_valid=bool(evaluations),
            evaluation_diagnostics=(
                dict(self.window.diagnostics)
                if not self.count_objective
                else {
                    "reason": "ready" if evaluations else "insufficient_fresh_episodes",
                    "fresh_episodes": sum(g["completed_episodes"] for g in evaluations.values()),
                }
            ),
            last_exploit_frame=self.last_exploit_frame,
            pbt_rng_state=pbt_utils.rng_state(self.rng),
            terminal=terminal,
            rollout_frames=int(getattr(self.algo, "batch_size_envs", 0)),
            selection_evidence=self.evidence,
            lineage=self.lineage,
            segment_rollouts=self.segment_rollouts,
            cold_first_19=self.segment_rollouts <= 19,
        )
        pbt_utils.cleanup({}, self.curr_policy_dir, max_yaml=3 if self.count_objective else 50)
        if terminal:
            self._report_selection(own, {"reason": "terminal", "donor": None})
            print(
                f"[pbt] final_checkpoint frame={current} epoch={self.algo.epoch_num} path={own['checkpoint']} "
                f"objective={self.cfg.objective} score={score if evaluations else None} "
                f"episodes={sum(g['completed_episodes'] for g in evaluations.values())}",
                flush=True,
            )
            return
        if not evaluations:
            self._report_selection(own, {"reason": "own_" + own["evaluation_diagnostics"]["reason"], "donor": None})
            return
        # Count objectives retain their disjoint-window semantics. DexPBT keeps
        # rolling evidence across publications; ready() prunes each old batch.
        if self.count_objective:
            self.window.consume()
        # Allegro compares fresh samples regardless of asynchronous lifetime age.
        latest = float("inf") if self.count_objective else iteration
        records = pbt_utils.load_pbt_ckpts(self.ws_dir, self.cfg.policy_idx, self.cfg.num_policies, latest)
        if self.cfg.objective == "allegro_goals":
            print(
                f"[pbt] publication member={self.cfg.policy_idx} frame={current} goals/episode={score:.6f} "
                f"episodes={sum(g['completed_episodes'] for g in evaluations.values())}",
                flush=True,
            )
        if self.cfg.objective == "contact_goals":
            stats = evaluations["native_contact"]
            print(
                f"[pbt] publication member={self.cfg.policy_idx} frame={current} "
                f"held_goals/episode={stats['goals_per_episode']:.6f} "
                f"touch_occupancy={stats['progress']:.6f} episodes={stats['completed_episodes']}",
                flush=True,
            )
        if self.cfg.objective == "grasp_hold":
            stats = evaluations["grasp_hold"]
            print(
                f"[pbt] publication member={self.cfg.policy_idx} frame={current} "
                f"safe_retention={stats['success_rate']:.6f} safe_valid4={stats['progress']:.6f} "
                f"overload_rate={stats['overload_rate']:.6f} episodes={stats['completed_episodes']}",
                flush=True,
            )
        if self.cfg.objective == "grasp_motion":
            stats = evaluations["grasp_motion"]
            print(
                f"[pbt] publication member={self.cfg.policy_idx} frame={current} "
                f"safe_tracking={stats['success_rate']:.6f} safe_retention={stats['progress']:.6f} "
                f"overload_rate={stats['overload_rate']:.6f} episodes={stats['completed_episodes']}",
                flush=True,
            )
        if current < self.cfg.warmup_steps:
            self._report_selection(own, {"reason": "warmup", "donor": None})
            return
        if current - self.last_exploit_frame < self.cfg.adaptation_steps:
            self._report_selection(own, {"reason": "adaptation", "donor": None})
            return
        if not self.cfg.selection_enabled:
            self._report_selection(own, {"reason": "selection_disabled", "donor": None})
            return
        if self.cfg.objective == "allegro_goals":
            from .allegro import select_goal_donor

            diagnostics = []
            donor_id = select_goal_donor(records, own, self.cfg, evidence=self.evidence, diagnostics=diagnostics)
            own["selection_evidence"] = self.evidence
            pbt_utils.atomic_yaml(own["pbt_checkpoint"], own)
            for detail in diagnostics:
                print(
                    "[pbt-selection] " + json.dumps({"member": self.cfg.policy_idx, "frame": current, **detail}),
                    flush=True,
                )
            print(f"[pbt] selection member={self.cfg.policy_idx} frame={current} donor={donor_id}", flush=True)
        elif self.cfg.objective == "contact_goals":
            from .contact import select_contact_donor

            donor_id = select_contact_donor(records, own, self.cfg)
            print(f"[pbt] selection member={self.cfg.policy_idx} frame={current} donor={donor_id}", flush=True)
        elif self.cfg.objective == "grasp_hold":
            from .grasp import select_grasp_donor

            donor_id = select_grasp_donor(records, own, self.cfg)
            print(f"[pbt] selection member={self.cfg.policy_idx} frame={current} donor={donor_id}", flush=True)
        elif self.cfg.objective == "grasp_motion":
            from .grasp_motion import select_motion_donor

            donor_id = select_motion_donor(records, own, self.cfg)
            print(f"[pbt] selection member={self.cfg.policy_idx} frame={current} donor={donor_id}", flush=True)
        else:
            diagnostics = {}
            donor_id = pbt_utils.select_donor(records, own, self.cfg, diagnostics=diagnostics)
            self._report_selection(own, diagnostics)
        if donor_id is None:
            return
        donor = records[donor_id]
        new_params = (
            dict(donor["params"])
            if self.cfg.copy_only
            else mutate(
                donor["params"],
                self.cfg.mutation,
                self.cfg.mutation_rate,
                self.cfg.change_range,
                self.cfg.mutation_bounds,
                rng=self.rng,
            )
        )
        # The normal publication precedes the tournament; restart must capture the
        # recipient stream AFTER mutation, not that publication or the donor stream.
        restart_recipient = {**own, "pbt_rng_state": pbt_utils.rng_state(self.rng)}
        try:
            checkpoint, metadata = pbt_utils.stage_restart(self.curr_policy_dir, donor, restart_recipient, new_params)
        except FileNotFoundError:
            print("[pbt] Donor pruned during copy; defer to the next fresh tournament", flush=True)
            return
        event = resume_metadata(checkpoint, metadata)
        print(
            "[pbt-replacement] "
            + json.dumps(
                {
                    "member": self.cfg.policy_idx,
                    "frame": current,
                    "donor": donor_id,
                    "lineage": event["lineage"],
                    "donor_lineage": event["donor_lineage"],
                    "recipient_previous_lineage": self.lineage,
                    "copy_only": self.cfg.copy_only,
                    "checkpoint_copy_seconds": event["checkpoint_copy_seconds"],
                    "changed_parameters": {
                        k: [donor["params"][k], v] for k, v in new_params.items() if donor["params"][k] != v
                    },
                }
            ),
            flush=True,
        )
        command = restart_command(self.original_argv, sys.executable, checkpoint, metadata, new_params)
        self.algo.writer.flush()
        self.algo.writer.close()
        print(
            f"[pbt] Member {self.cfg.policy_idx} exploits {donor_id} at recipient frame {current}: {command}",
            flush=True,
        )
        self.window.consume()
        self._rollout_start_frame = None
        self._rollout_completed_frames = 0
        os.execv(sys.executable, command)


class MultiObserver(AlgoObserver):
    """Compose standard Isaac, exploration, and PBT observer public callbacks."""

    def __init__(self, observers_):
        super().__init__()
        self.observers = observers_

    def _call_multi(self, method, *args, **kwargs):
        for observer in self.observers:
            getattr(observer, method)(*args, **kwargs)

    def before_init(self, base_name, config, experiment_name):
        self._call_multi("before_init", base_name, config, experiment_name)

    def after_init(self, algo):
        self._call_multi("after_init", algo)

    def process_infos(self, infos, done_indices):
        self._call_multi("process_infos", infos, done_indices)

    def after_steps(self):
        self._call_multi("after_steps")

    def after_clear_stats(self):
        self._call_multi("after_clear_stats")

    def after_print_stats(self, frame, epoch_num, total_time):
        self._call_multi("after_print_stats", frame, epoch_num, total_time)
