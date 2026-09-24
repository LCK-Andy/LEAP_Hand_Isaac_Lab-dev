"""Completed-episode asynchronous PBT with recipient-owned restart transactions."""

import os
import random
import sys

from rl_games.common.algo_observer import AlgoObserver

from . import pbt_utils
from .mutation import mutate
from .pbt_cfg import PbtCfg
from .runtime import PREFIX, SUPPORTED, apply_overrides, effective_params, install_restore, restart_command


def initialize_population(params, resumed=False):
    """Diversify new members before runner/observer construction; member zero is the baseline."""
    cfg = PbtCfg(**params["pbt"])
    values = pbt_utils.filter_params(pbt_utils.flatten_dict({"agent": params}), cfg.mutation)
    if cfg.enabled and cfg.initial_diversity and cfg.policy_idx != 0 and not resumed:
        values = mutate(
            values,
            cfg.mutation,
            1.0,
            cfg.change_range,
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

        self.window = GoalWindow() if self.cfg.objective == "allegro_goals" else pbt_utils.EpisodeWindow()
        self.pbt_it = None
        self.last_exploit_frame = (self.resume or {}).get("last_exploit_frame", 0)

    def after_init(self, algo):
        self.algo = algo
        self.ws_dir = os.path.join(algo.train_dir, self.cfg.workspace)
        self.curr_policy_dir = os.path.join(self.ws_dir, f"{self.cfg.policy_idx:03d}")
        if self.cfg.enabled:
            os.makedirs(self.curr_policy_dir, exist_ok=True)
        install_restore(algo, self.pbt_params, self.resume)
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
            self.window.add(infos, self.algo.frame)

    def after_steps(self):
        # rl-games calls after_steps BEFORE PPO updates and before incrementing frame.
        # Publish in after_print_stats instead so weights/optimizer and frame agree.
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        if not self.cfg.enabled:
            return
        current = int(self.algo.frame)
        iteration = current // self.cfg.interval_steps
        if self.pbt_it is None:
            restored_frame = (self.resume or {}).get("recipient_frame", (self.resume or {}).get("frame", 0))
            self.pbt_it = int(restored_frame) // self.cfg.interval_steps
        if iteration <= self.pbt_it:
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
            last_exploit_frame=self.last_exploit_frame,
            pbt_rng_state=pbt_utils.rng_state(self.rng),
        )
        pbt_utils.cleanup({}, self.curr_policy_dir, max_yaml=3 if self.cfg.objective == "allegro_goals" else 50)
        if not evaluations:
            return
        # Consume each local window once, never select repeatedly on stale episodes.
        self.window.consume()
        # Allegro compares fresh samples regardless of asynchronous lifetime age.
        latest = float("inf") if self.cfg.objective == "allegro_goals" else iteration
        records = pbt_utils.load_pbt_ckpts(self.ws_dir, self.cfg.policy_idx, self.cfg.num_policies, latest)
        if self.cfg.objective == "allegro_goals":
            print(
                f"[pbt] publication member={self.cfg.policy_idx} frame={current} goals/episode={score:.6f} "
                f"episodes={sum(g['completed_episodes'] for g in evaluations.values())}",
                flush=True,
            )
        if current < self.cfg.warmup_steps or current - self.last_exploit_frame < self.cfg.adaptation_steps:
            return
        if not self.cfg.selection_enabled:
            return
        if self.cfg.objective == "allegro_goals":
            from .allegro import select_goal_donor

            donor_id = select_goal_donor(records, own, self.cfg)
            print(f"[pbt] selection member={self.cfg.policy_idx} frame={current} donor={donor_id}", flush=True)
        else:
            donor_id = pbt_utils.select_donor(records, own, self.cfg)
        if donor_id is None:
            return
        donor = records[donor_id]
        new_params = mutate(
            donor["params"],
            self.cfg.mutation,
            self.cfg.mutation_rate,
            self.cfg.change_range,
            self.cfg.mutation_bounds,
            rng=self.rng,
        )
        # The normal publication precedes the tournament; restart must capture the
        # recipient stream AFTER mutation, not that publication or the donor stream.
        restart_recipient = {**own, "pbt_rng_state": pbt_utils.rng_state(self.rng)}
        try:
            checkpoint, metadata = pbt_utils.stage_restart(self.curr_policy_dir, donor, restart_recipient, new_params)
        except FileNotFoundError:
            print("[pbt] Donor pruned during copy; defer to the next fresh tournament", flush=True)
            return
        command = restart_command(self.original_argv, sys.executable, checkpoint, metadata, new_params)
        self.algo.writer.flush()
        self.algo.writer.close()
        print(
            f"[pbt] Member {self.cfg.policy_idx} exploits {donor_id} at recipient frame {current}: {command}",
            flush=True,
        )
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
