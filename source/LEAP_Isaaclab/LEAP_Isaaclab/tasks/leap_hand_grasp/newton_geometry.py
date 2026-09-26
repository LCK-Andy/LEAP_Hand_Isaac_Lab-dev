"""Stage-local LEAP collision schema relocation for Newton's USD importer.

The bundled hand authors CollisionAPI on instanceable ``collisions`` Xforms,
not their Cube/Mesh descendants. PhysX expands this convention, whereas Newton
1.5.2 ignores those Xforms and imports a hand with zero collision shapes.

This does not change points, topology, transforms, analytic dimensions, source
USD files, or installed packages. The authored convexDecomposition request is
preserved, but Newton implements it with CoACD, not PhysX's cooking algorithm.
The resulting dynamics therefore require independent validation.
"""


def prepare_newton_hand_collisions(stage, root: str = "/World/envs/env_0/Robot") -> dict:
    """Relocate inherited collision schemas onto geometry before Newton cloning.

    Call after spawning the hand but before cloning/model finalization. All edits
    go to the active stage's session layer; source assets are never saved. Only
    named collision subtrees are de-instanced; visual instances remain intact.

    Returns:
        Counts of converted groups, colliders, mesh approximations and instances.

    Raises:
        RuntimeError: If a collision group is empty, has ambiguous nested rigid
            bodies, or no collision groups are found under the requested robot.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    root_prim = stage.GetPrimAtPath(root)
    if not root_prim.IsValid():
        raise RuntimeError(f"Newton collision root does not exist: {root}")
    group_paths = [
        prim.GetPath()
        for prim in Usd.PrimRange(root_prim)
        if prim.GetName() == "collisions" and prim.IsA(UsdGeom.Xform)
    ]
    if not group_paths:
        raise RuntimeError(f"No LEAP collision groups found under {root}")
    report = {"groups": 0, "colliders": 0, "mesh_approximations": {}, "deinstanced": 0}
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in group_paths:
            group = stage.GetPrimAtPath(path)
            inherited_enabled = UsdPhysics.CollisionAPI(group).GetCollisionEnabledAttr().Get()
            approximation = group.GetAttribute("physics:approximation").Get()
            # Re-traverse after every de-instance; previously-hidden nested instances
            # may only become writable once their outer instance has been opened.
            while True:
                instances = [prim for prim in Usd.PrimRange(group) if prim.IsInstance()]
                if not instances:
                    break
                for prim in instances:
                    prim.SetInstanceable(False)
                    report["deinstanced"] += 1
            descendants = list(Usd.PrimRange(group))
            if any(prim != group and prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in descendants):
                raise RuntimeError(f"Unexpected nested rigid body under collision group {path}")
            geometry = [prim for prim in descendants if prim.IsA(UsdGeom.Gprim)]
            if not geometry:
                raise RuntimeError(f"Newton collision group has no geometry: {path}")
            for prim in geometry:
                had_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
                collision = UsdPhysics.CollisionAPI.Apply(prim)
                if not had_collision and inherited_enabled is not None:
                    collision.CreateCollisionEnabledAttr().Set(inherited_enabled)
                if prim.IsA(UsdGeom.Mesh):
                    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                    attr = mesh_collision.GetApproximationAttr()
                    if not attr.HasAuthoredValueOpinion() and approximation is not None:
                        mesh_collision.CreateApproximationAttr().Set(approximation)
                    name = str(mesh_collision.GetApproximationAttr().Get())
                    report["mesh_approximations"][name] = report["mesh_approximations"].get(name, 0) + 1
                report["colliders"] += 1
            for prim in descendants:
                if not prim.IsA(UsdGeom.Gprim):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
                    prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            report["groups"] += 1
    return report


def validate_newton_collision_coverage(model, sensors) -> dict:
    """Fail if a sensed/filtered body has no imported shape that can collide.

    Body and partner names can resolve successfully even when the USD importer
    discarded all their geometry. Run once after sensor/model initialization,
    never infer coverage from instantaneous force values in a no-contact state.
    """
    from newton import ShapeFlags

    required = set()
    for sensor in sensors:
        view = sensor.contact_view
        if view.sensing_type != "body" or view.counterpart_type != "body":
            raise RuntimeError("Stage A collision-coverage validation requires body-level sensor and partner filters.")
        required.update(view.sensing_indices)
        for row in view.counterpart_indices:
            required.update(row)
    if not required:
        raise RuntimeError("Newton collision-coverage validation found no sensing/partner bodies.")
    colliding = {}
    for body, flags in zip(model.shape_body.numpy().tolist(), model.shape_flags.numpy().tolist(), strict=True):
        if flags & int(ShapeFlags.COLLIDE_SHAPES):
            colliding[body] = colliding.get(body, 0) + 1
    missing = sorted(required - colliding.keys())
    if missing:
        names = [model.body_label[index] for index in missing]
        raise RuntimeError(
            f"Newton imported no collision-enabled geometry for {len(missing)} required sensing/partner bodies: "
            f"{names}. CollisionAPI must be on actual geometry, not only ancestor Xforms."
        )
    return {"required_bodies": len(required), "colliding_shapes": sum(colliding[index] for index in required)}
