#!/usr/bin/env python3
"""Convert the pinned released DexPBT robot, preserving original mesh and inertial sources.

Run in env_isaaclab with --source-root pointing to IsaacGymEnvs at SOURCE_COMMIT.
The source checkout is never modified. --preflight performs only CPU/filesystem checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE_COMMIT = "665c32170d84b4be66722eea405a1e08b6e7f761"
ROBOT_RELATIVE = "assets/urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf"
ROBOT_SHA256 = "52dc7ab9013ed46ddfa31ad5ff39df7365ad8698e6abe41e1480d762f47c7727"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "source/LEAP_Isaaclab/LEAP_Isaaclab/assets/dexpbt_original"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_source(source):
    """Verify the release and resolve both mesh reference conventions used by its URDF."""
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"Expected IsaacGymEnvs {SOURCE_COMMIT}, found {commit}")
    copied_scopes = ["assets/urdf/kuka_allegro_description", "assets/licenses", "LICENSE.txt"]
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all", "--", *copied_scopes],
        text=True,
    )
    if dirty.strip():
        raise ValueError(f"Source asset scope differs from the pinned checkout (including textures):\n{dirty}")
    urdf = source / ROBOT_RELATIVE
    if digest(urdf) != ROBOT_SHA256:
        raise ValueError("Original robot URDF differs from the pinned release")
    tree = ET.parse(urdf)
    mesh_paths = {}
    for mesh in tree.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        candidates = [urdf.parent / filename, source / "assets/urdf" / filename]
        existing = {path.resolve() for path in candidates if path.is_file()}
        if len(existing) != 1:
            raise ValueError(f"Mesh must resolve unambiguously: {filename}: {existing}")
        mesh_paths[filename] = existing.pop()
    joints = []
    for joint in tree.getroot().findall("joint"):
        if joint.attrib["type"] == "revolute":
            joints.append(
                {
                    "name": joint.attrib["name"],
                    "limits": joint.find("limit").attrib,
                    "dynamics": joint.find("dynamics").attrib,
                }
            )
    if len(joints) != 23:
        raise ValueError(f"Expected 23 revolute joints, found {len(joints)}")
    return (
        tree,
        mesh_paths,
        {
            "source_repository": "https://github.com/isaac-sim/IsaacGymEnvs",
            "source_commit": commit,
            "source_urdf": ROBOT_RELATIVE,
            "source_urdf_sha256": digest(urdf),
            "joint_contract": joints,
            "source_link_count": len(tree.getroot().findall("link")),
            "source_total_mass": sum(float(node.attrib["value"]) for node in tree.findall(".//mass")),
            "mesh_sha256": {str(path.relative_to(source)): digest(path) for path in mesh_paths.values()},
            "limitations": [
                "Source finger inertia tensors include non-positive eigenvalues; no source repair is performed.",
                "USD conversion and modern PhysX cooking are not proof of Isaac Gym dynamic equivalence.",
                "URDF default self-collision pair exclusions require simulator validation.",
            ],
        },
    )


def prepare_import(source, output, tree, mesh_paths):
    """Copy licensed source dependencies; rewrite mesh paths only in the import copy."""
    copied = output / "source"
    copied.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / ROBOT_RELATIVE, copied / "original.urdf")
    shutil.copytree(source / "assets/licenses", copied / "licenses", dirs_exist_ok=True)
    shutil.copy2(source / "LICENSE.txt", copied / "LICENSE.txt")
    # Copy complete mesh folders: OBJ material and texture references remain relative and portable.
    shutil.copytree((source / ROBOT_RELATIVE).parent / "meshes", copied / "meshes", dirs_exist_ok=True)
    for mesh in tree.findall(".//mesh"):
        resolved = mesh_paths[mesh.attrib["filename"]]
        relative = resolved.relative_to((source / ROBOT_RELATIVE).parent)
        mesh.attrib["filename"] = str((copied / relative).resolve())
    normalized = copied / "kuka_allegro_touch_sensor.urdf"
    tree.write(normalized, encoding="utf-8", xml_declaration=True)
    return normalized


def remove_importer_world_joint(stage):
    """Remove only the importer's non-body world anchor; spawn authors the canonical weld."""
    from pxr import UsdPhysics

    removed = []
    for prim in list(stage.Traverse()):
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if len(body0) != 1 or len(body1) != 1:
            continue
        first = stage.GetPrimAtPath(body0[0])
        second = stage.GetPrimAtPath(body1[0])
        if not first.HasAPI(UsdPhysics.RigidBodyAPI) and second.HasAPI(UsdPhysics.ArticulationRootAPI):
            removed.append(str(prim.GetPath()))
            stage.RemovePrim(prim.GetPath())
    if len(removed) != 1:
        raise RuntimeError(f"Expected exactly one malformed importer world joint, got {removed}")
    return removed


def strip_root_mirror(stage):
    """Remove only the importer backend mirror which blocks EA referenced-root migration."""
    from pxr import UsdPhysics

    removed = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            name = "newton:selfCollisionEnabled"
            prop = prim.GetProperty(name)
            if prop and prop.IsAuthored():
                if not prim.RemoveProperty(name):
                    raise RuntimeError(f"Cannot remove root mirror {prim.GetPath()}.{name}")
                removed.append(f"{prim.GetPath()}.{name}")
            if "NewtonArticulationRootAPI" in prim.GetAppliedSchemas():
                prim.RemoveAppliedSchema("NewtonArticulationRootAPI")
    return removed


def repair_root_mirror(output):
    from pxr import Usd

    asset = output / "robot.usd"
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_commit"] != SOURCE_COMMIT:
        raise ValueError("Refusing to repair an asset from a different source")
    backup = output / "robot.importer-root-mirror.usd"
    if backup.exists():
        raise FileExistsError(f"Preserved original already exists: {backup}")
    stage = Usd.Stage.Open(str(asset))
    removed = strip_root_mirror(stage)
    if not removed:
        raise RuntimeError("No authored root mirror to repair")
    shutil.copy2(asset, backup)
    stage.Export(str(asset))
    manifest["converted"]["removed_root_mirrors"] = removed
    manifest["converted"]["sha256"] = digest(asset)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Removed {removed}; original preserved at {backup}")


def repair_existing(output):
    """Repair an already-built asset without re-running the importer or changing inertias."""
    from pxr import Usd

    asset = output / "robot.usd"
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_commit"] != SOURCE_COMMIT:
        raise ValueError("Refusing to repair an asset from a different source")
    backup = output / "robot.importer-world-joint.usd"
    if backup.exists():
        raise FileExistsError(f"Preserved original already exists: {backup}")
    stage = Usd.Stage.Open(str(asset))
    removed = remove_importer_world_joint(stage)
    shutil.copy2(asset, backup)
    stage.Export(str(asset))
    manifest["converted"]["removed_importer_world_joints"] = removed
    manifest["converted"]["requires_spawn_fix_root_link"] = True
    manifest["converted"]["sha256"] = digest(asset)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Removed {removed}; original preserved at {backup}")


def convert_robot(normalized, output, manifest):
    """Use EA importer 3.0 without legacy regex surgery or inertia substitutions."""
    from pxr import Usd, UsdPhysics

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(normalized),
            usd_dir=str(output / "converted"),
            fix_base=True,
            merge_fixed_joints=True,
            force_usd_conversion=True,
            make_instanceable=False,
            link_density=0.0,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=40.0, damping=5.0),
            ),
            collision_from_visuals=False,
            collision_type="Convex Hull",
            self_collision=True,
            merge_mesh=False,
            run_asset_transformer=False,
            run_multi_physics_conversion=False,
        )
    )
    stage = Usd.Stage.Open(converter.usd_path)
    if stage is None:
        raise RuntimeError(f"Cannot open generated asset: {converter.usd_path}")
    removed_world_joints = remove_importer_world_joint(stage)
    removed_root_mirrors = strip_root_mirror(stage)
    roots, bodies, joints, colliders = [], [], [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            bodies.append(
                {
                    "path": str(prim.GetPath()),
                    "mass": mass_api.GetMassAttr().Get(),
                    "diagonal_inertia": str(mass_api.GetDiagonalInertiaAttr().Get()),
                }
            )
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joints.append(prim.GetName())
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(str(prim.GetPath()))
    expected = {joint["name"] for joint in manifest["joint_contract"]}
    if len(roots) != 1 or set(joints) != expected or not colliders:
        raise RuntimeError(f"Conversion contract failed: roots={roots}, joints={joints}, colliders={len(colliders)}")
    asset_path = output / "robot.usd"
    stage.Export(str(asset_path))
    manifest["converted"] = {
        "usd": "robot.usd",
        "sha256": digest(asset_path),
        "articulation_roots": roots,
        "bodies": bodies,
        "colliders": colliders,
        "revolute_joints": joints,
        "removed_importer_world_joints": removed_world_joints,
        "removed_root_mirrors": removed_root_mirrors,
        "requires_spawn_fix_root_link": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["converted"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", action="store_true", help="Validate source only; do not import or launch Kit")
    parser.add_argument("--repair-existing", action="store_true", help="Repair imported world joint using USD only")
    parser.add_argument("--strip-root-mirror", action="store_true", help="Repair referenced root mirror using USD only")
    # Parse preflight without importing Isaac Lab, which keeps this path simulator-free.
    args, _ = parser.parse_known_args()
    source = args.source_root.resolve()
    tree, mesh_paths, manifest = inspect_source(source)
    if args.preflight:
        print(json.dumps(manifest, indent=2))
        return 0
    if args.repair_existing:
        repair_existing(args.output.resolve())
        return 0
    if args.strip_root_mirror:
        repair_root_mirror(args.output.resolve())
        return 0
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    output = args.output.resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite a completed asset build: {output}")
    normalized = prepare_import(source, output, tree, mesh_paths)
    launcher = AppLauncher(args)
    exit_code = 1
    try:
        convert_robot(normalized, output, manifest)
        exit_code = 0
    finally:
        launcher.app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
