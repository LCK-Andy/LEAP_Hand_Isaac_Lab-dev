# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Objects variant: the direct in-hand reorientation task (v0) generalized to a
# catalog of different objects (YCB and others). Isaac Lab 3.0.0b2's
# heterogeneous clone plan (MultiUsdFileCfg in the scene cfg) is broken for the
# DirectRLEnv flow, so each env's object is spawned individually after cloning
# (one randomly chosen catalog USD per env; the assignment is fixed for the
# lifetime of the env). The official MDP is inherited unchanged from v0.
# --------------------------------------------------------

from isaaclab.utils.configclass import configclass

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from LEAP_Isaaclab.tasks.leap_hand_reorient.leap_hand_reorient_ycb_env_cfg import _OBJECTS

from .leap_direct_env_cfg import LeapHandDirectEnvCfg

# catalog subset with locally bundled USD wrappers (see scripts/download_ycb.py
# and scripts/convert_ycb_to_usd.py to regenerate them)
_OBJECT_NAMES = [
    # boxes
    "cracker_box",
    "sugar_box",
    "pudding_box",
    "gelatin_box",
    # cans and bottles
    "tomato_soup_can",
    "mustard_bottle",
    "tuna_fish_can",
    "potted_meat_can",
    "master_chef_can",
    # fruits
    "banana",
    "strawberry",
    "apple",
    "lemon",
    "peach",
    "pear",
    "orange",
    # others
    "wood_block",
    "foam_brick",
    "rubiks_cube",
]

_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=0,
    sleep_threshold=0.005,
    stabilization_threshold=0.0025,
    max_depenetration_velocity=1000.0,
)


def object_spawn_cfg(usd_path: str) -> sim_utils.UsdFileCfg:
    """Spawn cfg for one catalog object (shared rigid properties).

    NOTE: mass_props is intentionally omitted -- the YCB wrapper USDs carry no mass
    attributes, so density set here is silently ignored; the env authors the USD
    Physics MassAPI density on each template prim instead (``object_density``).
    """
    return sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=_RIGID_PROPS,
    )


@configclass
class LeapObjectsDirectEnvCfg(LeapHandDirectEnvCfg):
    """Direct in-hand reorientation over a catalog of palm-size objects.

    Identical MDP to ``Isaac-Repose-Cube-LEAP-Direct-v0`` (124-d "full" obs,
    16-d actions, rewards, resets) except the in-hand object is resampled every
    episode from the candidate ``object_names`` catalog: objects whose measured
    USD extent exceeds ``max_object_extent_m`` are dropped, the rest are all
    spawned per env (spawn-all-and-toggle), and each episode reset randomly
    activates one per env. Object mass varies with density (400 kg/m^3) x mesh
    volume. Objects spawn slightly higher (z=0.62) to clear the opened fingers.
    """

    object_names: list = _OBJECT_NAMES
    max_object_extent_m = 0.09  # LEAP palm is ~9 cm across; larger objects are dropped
    object_density = 400.0  # kg/m^3, authored via USD Physics MassAPI in the env
    # collision geometry: "hull" (default) uses the wrapper USDs directly --
    # one convex hull per object (the env de-duplicates the wrappers' double
    # CollisionAPI; measured 3.3x faster and -1.8 GiB vs 14 spheres at 1024
    # envs). "spheres" uses MorphIt-derived
    # sphere-proxy USDs (assets/ycb/spheres/<name>_spheres.usd, see
    # scripts/generate_sphere_proxy_usd.py) -- no concavity error, tighter mass
    collision_proxy = "hull"
    # sphere-proxy USD dir override (e.g. the 50-sphere fits in
    # assets/ycb/spheres50); None -> assets/ycb/spheres (14-sphere fits)
    sphere_proxy_dir: str | None = None
    # goal orientation triad (official frame_prim arrows) scale [m]; the native
    # widget is ~1 m across, palm-size objects warrant a small frame
    goal_axes_scale = 0.12

    def __post_init__(self):
        super().__post_init__()
        # this task spawns the whole catalog per env (8 rigid bodies vs the cube
        # task's 1). This Isaac Lab beta never applies the PhysX GPU buffer cfg
        # fields, so counts >~1024 envs overflow the default patch buffer
        # ("Patch buffer overflow" errors, degraded physics). Default to 1024;
        # override with --num_envs on hardware/builbs where the buffers work.
        self.scene.num_envs = 1024
        # 8 rigid bodies per env (the parked catalog) exhausts the official patch
        # budget at large env counts (PhysX "Patch buffer overflow" corrupts the
        # sim with exploding rewards) -- scale the GPU buffers up. NOTE: sim.physics
        # is the repo's PresetCfg; the buffer fields live on its `physx` variant.
        physx = getattr(self.sim.physics, "physx", None)
        if physx is not None and hasattr(physx, "gpu_max_rigid_patch_count"):
            physx.gpu_max_rigid_patch_count = 2**22
            physx.gpu_max_rigid_contact_count = 2**24
