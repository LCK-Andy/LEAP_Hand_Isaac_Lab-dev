#!/usr/bin/env python3
"""Query pinned DexPBT assets with original Gym on CPU, without stepping or rendering.

Use a Python version supported by the supplied SDK (Preview4: 3.6-3.8). Set
CUDA_VISIBLE_DEVICES='' and any needed loader paths in the calling process.
No packages, source assets, or installed configuration are modified. Output must
not already exist. --compare-usd instead uses simulator-free pxr and NumPy to
compare authored USD properties against a measured fixture; it does not certify
runtime PhysX properties or dynamic equivalence.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE_COMMIT = "665c32170d84b4be66722eea405a1e08b6e7f761"
ROBOT = "urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf"
ROBOT_SHA = "52dc7ab9013ed46ddfa31ad5ff39df7365ad8698e6abe41e1480d762f47c7727"
SHAPE_FIELDS = (
    "friction",
    "rolling_friction",
    "torsion_friction",
    "restitution",
    "compliance",
    "filter",
    "contact_offset",
    "rest_offset",
    "thickness",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_fixture(raw):
    """Normalize raw no-step oracle output into a measured, non-upstream fixture."""
    robot = {
        "source": ROBOT,
        "source_sha256": raw["source_sha256"],
        "asset_options": raw["original_asset_options"],
        "body_shape_ranges": raw["body_shape_ranges"],
        "bodies": raw["actor_bodies"],
        "asset_shapes": raw["asset_shapes"],
        "actor_shapes": raw["actor_shapes"],
    }
    assets = {"robot": robot}
    for name, extra in raw["additional_assets"].items():
        assets[name] = {key: extra[key] for key in ("source_sha256", "bodies", "asset_shapes", "actor_shapes")}
        assets[name]["source"] = "urdf/table_narrow.urdf" if name == "table" else "generated/000_cube_100_100_100.urdf"
        assets[name]["asset_options"] = {"fix_base_link": True, "disable_gravity": False} if name == "table" else {}
        if name == "default_cube":
            assets[name]["generator_sha256"] = extra["generator_sha256"]
            assets[name]["dimensions_m"] = [0.05, 0.05, 0.05]
            assets[name]["density_kg_m3"] = 400.0
            # Rendered XML contains an absolute visual-mesh path, so retain it for hash provenance.
            assets[name]["generated_source_text"] = extra["source_text"]
    return {
        "schema_version": 1,
        "provenance": {
            "origin": "Measured original Isaac Gym Preview4 CPU PhysX actor properties; NOT an upstream source manifest",
            "source_repository": "https://github.com/isaac-sim/IsaacGymEnvs",
            "source_commit": SOURCE_COMMIT,
            "sdk_sha256": raw["sdk_sha256"],
            "python": raw["python"],
            "compute_device": -1,
            "graphics_device": -1,
            "use_gpu_pipeline": False,
            "physx_use_gpu": False,
            "simulation_steps": 0,
            "viewer_created": False,
            "robot_collision_filter": -1,
            "aggregate_self_collision": True,
            "coordinates": "COM and full symmetric inertia about COM in imported body coordinates; SI units",
            "friction_combine": "average, documented in SDK Preview4 release-notes.html; not a contact measurement",
            "limits": [
                "No dynamic or GPU-pipeline parity claim; negative finger inertia eigenvalues remain present.",
                "Shape filter zero does not prove absence of intrinsic articulation parent-child exclusions.",
                "Actor thickness getter returns zero although robot asset thickness is 0.001.",
            ],
        },
        "asset_defaults": raw["asset_defaults"],
        "assets": assets,
    }


def query(args):
    """Import original SDK only after validating CPU-only launch and all paths."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' before starting this process")
    source = args.source_root.resolve()
    sdk = args.sdk_root.resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT or digest(source / "assets" / ROBOT) != ROBOT_SHA:
        raise ValueError("Source checkout does not match the pinned original release")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(sdk / "python"))
    import numpy as np

    from isaacgym import gymapi

    if Path(gymapi.__file__).resolve().parent != sdk / "python/isaacgym":
        raise RuntimeError("Wrong Isaac Gym SDK imported")
    bindings = sdk / "python/isaacgym/_bindings/linux-x86_64"
    module_name = "gym_{}{}.so".format(*sys.version_info[:2])

    def vector(value):
        return [float(value.x), float(value.y), float(value.z)]

    def matrix(value):
        return np.array([vector(value.x), vector(value.y), vector(value.z)]).T.tolist()

    def shape(value):
        return {key: getattr(value, key) for key in SHAPE_FIELDS if hasattr(value, key)}

    def inspect(asset, options, label, group, collision_filter):
        root, filename = (source / "assets", ROBOT) if label == "robot" else (asset.parent, asset.name)
        loaded = gym.load_asset(sim, str(root), filename, options)
        if loaded is None:
            raise RuntimeError("Could not load " + str(asset))
        env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
        gym.begin_aggregate(env, gym.get_asset_rigid_body_count(loaded), gym.get_asset_rigid_shape_count(loaded), True)
        pose = gymapi.Transform()
        if label == "robot":
            pose.p = gymapi.Vec3(0, 0.8, 0)
        actor = gym.create_actor(env, loaded, pose, label, group, collision_filter, 0)
        if actor < 0:
            raise RuntimeError("Could not create " + label)
        gym.end_aggregate(env)
        bodies = []
        for name, item in zip(
            gym.get_actor_rigid_body_names(env, actor), gym.get_actor_rigid_body_properties(env, actor)
        ):
            inertia = matrix(item.inertia)
            bodies.append(
                dict(
                    name=name,
                    mass=item.mass,
                    invMass=item.invMass,
                    com={"p": vector(item.com)},
                    inertia=inertia,
                    invInertia=matrix(item.invInertia),
                    eigenvalues=np.linalg.eigvalsh(inertia).tolist(),
                    flags=item.flags,
                )
            )
        return {
            "source_sha256": digest(asset),
            "source_text": asset.read_text(),
            "bodies": bodies,
            "asset_shapes": [shape(item) for item in gym.get_asset_rigid_shape_properties(loaded)],
            "actor_shapes": [shape(item) for item in gym.get_actor_rigid_shape_properties(env, actor)],
            "body_shape_ranges": [
                dict(name=name, start=item.start, count=item.count)
                for name, item in zip(
                    gym.get_asset_rigid_body_names(loaded), gym.get_asset_rigid_body_shape_indices(loaded)
                )
            ],
        }

    options = gymapi.AssetOptions()
    raw = {
        "python": sys.version,
        "sdk_sha256": {name: digest(bindings / name) for name in (module_name, "libcarb.gym.plugin.so")},
    }
    raw["asset_defaults"] = {
        key: getattr(options, key)
        for key in (
            "override_inertia",
            "override_com",
            "density",
            "armature",
            "use_physx_armature",
            "vhacd_enabled",
            "convex_decomposition_from_submeshes",
            "replace_cylinder_with_capsule",
            "enable_gyroscopic_forces",
        )
    }
    original_options = dict(
        fix_base_link=True,
        flip_visual_attachments=False,
        collapse_fixed_joints=True,
        disable_gravity=True,
        thickness=0.001,
        angular_damping=0.01,
        linear_damping=0.01,
        use_physx_armature=True,
        default_dof_drive_mode=int(gymapi.DOF_MODE_POS),
    )
    for key, value in original_options.items():
        setattr(options, key, value)
    params = gymapi.SimParams()
    params.use_gpu_pipeline = False
    params.physx.use_gpu = False
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.dt, params.substeps = 0.01667, 2
    params.physx.num_threads = 1
    params.physx.solver_type = 1
    params.physx.num_position_iterations = 8
    params.physx.num_velocity_iterations = 0
    params.physx.contact_offset, params.physx.rest_offset = 0.002, 0.0
    gym = gymapi.acquire_gym()
    sim = gym.create_sim(-1, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("CPU PhysX context creation failed")
    try:
        # Use the original asset root for robot mesh references (two relative-path conventions).
        robot_options = options
        robot = inspect(source / "assets" / ROBOT, robot_options, "robot", 0, -1)
        raw.update(
            source_sha256=robot["source_sha256"],
            original_asset_options=original_options,
            body_shape_ranges=robot["body_shape_ranges"],
            actor_bodies=robot["bodies"],
            asset_shapes=robot["asset_shapes"],
            actor_shapes=robot["actor_shapes"],
        )
        table_options = gymapi.AssetOptions()
        table_options.fix_base_link, table_options.disable_gravity = True, False
        table = inspect(source / "assets/urdf/table_narrow.urdf", table_options, "table", 1, 0)
        generator_path = source / "isaacgymenvs/tasks/allegro_kuka/generate_cuboids.py"
        spec = importlib.util.spec_from_file_location("original_cube_generator", generator_path)
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        with tempfile.TemporaryDirectory(prefix="dexpbt-cpu-oracle-") as temporary:
            generator.generate_default_cube(
                temporary, str(source / "assets/urdf/objects/meshes/cube_multicolor.obj"), 0.05
            )
            cube = inspect(Path(temporary) / "000_cube_100_100_100.urdf", gymapi.AssetOptions(), "default_cube", 2, 0)
        cube["generator_sha256"] = digest(generator_path)
        raw["additional_assets"] = {"table": table, "default_cube": cube}
        return canonical_fixture(raw)
    finally:
        gym.destroy_sim(sim)


def compare_usd(usd_path, fixture_path):
    """Compare authored COM and R diag(I) R^T; Gf uses row-vector rotations."""
    import numpy as np

    from pxr import Gf, Usd, UsdPhysics

    fixture = json.loads(fixture_path.read_text())
    expected = {body["name"]: body for body in fixture["assets"]["robot"]["bodies"]}
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError("Cannot read USD")
    comparisons = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        body = expected.pop(prim.GetName(), None)
        if body is None:
            comparisons.append({"path": str(prim.GetPath()), "error": "Unexpected body"})
            continue
        api = UsdPhysics.MassAPI(prim)
        center = np.array(api.GetCenterOfMassAttr().Get(), dtype=float)
        rotation = np.array(Gf.Matrix3d(Gf.Quatd(api.GetPrincipalAxesAttr().Get()))).T
        inertia = rotation @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ rotation.T
        finite = bool(np.isfinite(center).all())
        comparisons.append(
            {
                "name": prim.GetName(),
                "path": str(prim.GetPath()),
                "mass_abs_error_kg": abs(api.GetMassAttr().Get() - body["mass"]),
                "com_authored_finite": finite,
                "com_max_abs_error_m": float(np.max(np.abs(center - body["com"]["p"]))) if finite else None,
                "inertia_max_abs_error_kg_m2": float(np.max(np.abs(inertia - body["inertia"]))),
            }
        )
    return {
        "scope": "Authored USD only, not runtime physics",
        "usd_sha256": digest(usd_path),
        "fixture_sha256": digest(fixture_path),
        "bodies": comparisons,
        "missing_bodies": sorted(expected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-usd", type=Path)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.compare_usd:
        if args.fixture is None:
            parser.error("--compare-usd requires --fixture")
        result = compare_usd(args.compare_usd, args.fixture)
    else:
        if args.sdk_root is None or args.source_root is None:
            parser.error("Query requires --sdk-root and --source-root")
        result = query(args)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
