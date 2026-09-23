# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Contact-point dexterity benchmark (see leap_contact_direct_env.py): the
# repose direct task with the orientation goal replaced by 4 cube-surface
# contact points. Inherited reward knobs that no longer apply (rot_*,
# fingertip_dist_*, palm_*, goal_press_*, fingertip_contact_*) are left at
# their parent values and simply unused; the inherited goal_rot obs channels
# stay at identity (fixed-reference orientation encoding).
# --------------------------------------------------------

from isaaclab.utils.configclass import configclass

from .leap_direct_env_cfg import LeapHandDirectEnvCfg


@configclass
class LeapContactDirectEnvCfg(LeapHandDirectEnvCfg):
    """Direct (DirectRLEnv) touch-4-contact-points task with the LEAP hand.

    4 points are sampled uniformly on the cube surface (all 6 faces, minimum
    pairwise separation); a pulp within ``touch_distance`` of a point AND
    pressing above ``touch_force`` counts as touching it. All 4 must be touched
    continuously for a per-env duration drawn from ``hold_time_range`` to score
    (``reach_goal_bonus``) and resample the points. PhysX-only.
    """

    observation_space = 184  # 136 repose obs + 4x4 tip->point relative vectors

    # --- contact-point task knobs ---
    cube_half_extent = 0.03  # measured DexCube bbox at scale 1.2 (6 cm cube); smoke re-checks this
    touch_distance = 0.015  # pulp-center-to-point distance gate [m]
    touch_force = 0.1  # cube-filtered pulp contact-force gate [N]; <= 0 = geometry-only mode
    # (kit-less newton fallback: sensor forces read zero there, so the gate drops out entirely)
    min_point_separation = 0.025  # minimum pairwise distance between sampled points [m] (> pulp diameter,
    # comfortably below the rejection-sampling failure regime at this cube size)
    max_point_resamples = 50  # rejection-resample cap for the separation constraint
    hold_time_range = (0.1, 0.5)  # per-env, per-goal hold duration sampled from this range [s];
    # 3-15 control steps -- must fit inside realistic episodes or the +250 is never observed
    point_dist_reward_scale = -0.05  # v3: matched pulp->point distance cost stream. Keep the
    # per-step cost UNDER the drop-exploit break-even 25/300 ~= 0.083: any larger stream
    # (tried -5.0 and -1.0) makes dropping early optimal once L x cost > fall_penalty, and
    # both the GRU-PPO and FlashSAC runs found that exactly (episode length boom -> bust).
    # Reaching is pulled by the positive bonuses (touch/alive/success), not by penalties.
    touch_bonus_scale = 0.5  # per touched point per step (max +2.0/step)
    # v3 positive-sum retention (2026-09-21): v1 had no retention gradient at all and v2's
    # negative streams still lost to the bounded fall cost. Holding must EARN.
    alive_bonus_scale = 0.1  # + per step while the cube is within alive_distance of the anchor
    alive_distance = 0.10  # alive-bonus radius around the in-hand anchor [m]
    cube_pull_scale = -0.25  # continuous drift direction signal; max -0.06/step inside the
    # fall radius (0.24 m), safely under the break-even
    reset_rot_scale = 0.0  # spawn-rotation magnitude: 0 = axis-aligned cube (identity); raise
    # toward 1.0 as a curriculum once retention is learned (1.0 = the official X/Y randomization)
    fall_penalty = -50.0  # v3: terminal cost on the out-of-reach termination (base has 0) --
    # doubling v2's -25 adds margin on top of the positive-sum alive bonus
    contact_marker_radius = 0.008  # debug sphere marker radius [m]

    def __post_init__(self):
        # large-scale training (>= 16k envs): the default PhysX GPU scene
        # buffers overflow around ~12k envs for this hand+cube contact load
        # (~250k contact patches at 16384 envs vs the default cap 163840) and
        # DROPPED patches silently corrupt the force-gated touch test. 2**19
        # patches / 2**24 contacts = ~4x the 8192-env demand.
        self.sim.physics.gpu_max_rigid_patch_count = 2**19
        self.sim.physics.gpu_max_rigid_contact_count = 2**24


@configclass
class LeapContactDirectEnvCfg_PLAY(LeapContactDirectEnvCfg):
    """Evaluation variant: fewer envs and no timeout truncation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        # DirectRLEnv derives the timeout from episode_length_s -- a huge value
        # effectively disables it for evaluation (the manager-based Play cfg
        # sets terminations.time_out = None; direct envs have no such knob).
        self.episode_length_s = 1.0e6
