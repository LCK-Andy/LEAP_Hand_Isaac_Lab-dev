# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Objects v1 cfg: the v0 catalog task with manifest-backed inertials. The MDP
# (obs/actions/rewards) is inherited unchanged; only the physics of the in-hand
# object differs -- see leap_objects_direct_v1_env.py.
# --------------------------------------------------------

from isaaclab.utils.configclass import configclass

from .leap_objects_direct_env_cfg import LeapObjectsDirectEnvCfg


@configclass
class LeapObjectsDirectV1EnvCfg(LeapObjectsDirectEnvCfg):
    """Catalog task with per-episode (mass, COM, inertia) from the ground-truth manifest.

    Identical MDP to ``Isaac-Repose-Objects-LEAP-Direct-v0`` except the object's
    inertial properties: every template prim is authored the full MassAPI state
    and each episode reset resamples one of the object's manifest variants
    (one per random density strategy of the sibling Estimate_weight_distribution
    dataset) via the RigidObjectCollection runtime writers. The ground truth is
    NOT observed by the policy (136-d obs unchanged). The manifest's synthetic
    densities (200-5000 kg/m^3) make objects much heavier than v0's uniform
    400 kg/m^3 -- up to ~0.9 kg for orange/pear -- a deliberate regime change
    for the inertial-estimation research line.

    ``object_density`` is inherited but unused (explicit mass overrides it).
    The manifest's per-object friction pair is not yet applied.
    """

    # manifest-covered objects with a bundled wrapper USD (plum has manifest
    # data but no USD in this repo; the other candidates -- tuna can, apple,
    # peach, foam brick, Rubik's cube -- have no manifest data yet)
    object_names: list = [
        "strawberry",
        "lemon",
        "orange",
        "pear",
    ]
    # path to the baked manifest; None -> assets/ycb/inertial_manifest.json
    inertial_manifest: str | None = None
