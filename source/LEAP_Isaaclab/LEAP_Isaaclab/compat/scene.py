"""Explicit EA cloning for direct tasks that spawn their template in _setup_scene."""

from isaaclab import cloner


def clone_environments(scene, global_prim_paths=("/World/ground",)):
    """Replicate the env_0 template and isolate PhysX environments, as native EA tasks do."""
    positions = cloner.grid_transforms(scene.num_envs, scene.cfg.env_spacing, device=scene.device)[0]
    plan = cloner.clone_plan_from_env_0(
        "/World/envs/env_0",
        "/World/envs/env_{}",
        scene.num_envs,
        scene.device,
        positions,
        global_paths=tuple(global_prim_paths),
    )
    cloner.replicate(plan, stage=scene.stage)
    if "physx" in scene.physics_backend:
        scene.filter_collisions(global_prim_paths=list(global_prim_paths))
