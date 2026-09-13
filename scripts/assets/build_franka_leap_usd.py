#!/usr/bin/env python3
"""Build franka_leap.usda -- a pre-built, fixed-base USD of the Franka+LEAP robot.

Pipeline (same philosophy as leap_hand_v1_right/, the pre-built LEAP hand USD):

  1. Convert franka_leap.urdf with the classic Isaac Lab UrdfConverter
     (asset-transformer / multi-physics conversion DISABLED -- that pipeline
     places the articulation root nondeterministically and emits a world
     root_joint the env cloner corrupts for envs > 0, leaving a floating base).
  2. Patch the converted file's articulation encoding (TEXT-level: a live
     Isaac Sim stage listener re-applies ArticulationRootAPI on any USD-API
     removal, so runtime/USD-API surgery is unreliable) to the structure of
     the working leap_hand USD: exactly one ArticulationRootAPI, on the BASE
     BODY (panda_link0), and no fixed joints anchored to non-body prims.
     The task cfg then sets fix_root_link=True, and Isaac Lab creates its
     world fixed joint (the cloner-safe shape) at spawn -- exactly like the
     LEAP hand asset.
  3. Save to assets/franka_leap/franka_leap.usda (self-contained flat file).

Run: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
     __NV_PRIME_RENDER_OFFLOAD=1 \
     ~/env_isaaclab/bin/python scripts/build_franka_leap_usd.py
"""

import argparse
import re
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build the Franka+LEAP pre-built USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / "source/LEAP_Isaaclab/LEAP_Isaaclab/assets"
URDF = ASSETS / "urdf/franka_leap.urdf"
OUT_DIR = ASSETS / "franka_leap"
OUT_FILE = OUT_DIR / "franka_leap.usd"
BUILD_DIR = ASSETS / "urdf/franka_leap_build"
BASE_LINK = "panda_link0"

# conversion settings mirror the spawn cfg previously used by FRANKA_LEAP_CFG;
# the runtime asset no longer converts from URDF -- it consumes the built USD.
CONVERTER_CFG = UrdfConverterCfg(
    asset_path=str(URDF),
    usd_dir=str(BUILD_DIR),
    usd_file_name="franka_leap.usd",
    fix_base=True,
    # merge fixed-joint links into their parents: otherwise link8/base/palm_lower
    # exist only as visual prims (PhysX fuses them), never get driven at runtime,
    # and the whole hand renders frozen at its authored pose
    merge_fixed_joints=True,
    force_usd_conversion=True,
    make_instanceable=False,
    link_density=0.0,
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
    ),
    collision_from_visuals=False,
    collision_type="convex_hull",
    self_collision=False,
    replace_cylinders_with_capsules=False,
    merge_mesh=False,
    robot_type="Default",
    run_asset_transformer=False,
    run_multi_physics_conversion=False,
    debug_mode=False,
)


def find_block(text: str, start: int) -> tuple[int, int]:
    """Return (start, end) of the brace-delimited block beginning at line `start`."""
    open_idx = text.index("{", text.index("\n", start))
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
    raise ValueError("unbalanced braces")


def patch_usda_text(text: str) -> str:
    """Re-encode the articulation as fixed-base (leap_hand USD structure)."""
    # bodies = prims carrying PhysicsRigidBodyAPI (for joint target validation)
    bodies = set(re.findall(r'def \w+ "([\w/]+)" \(\s*(?:#[^\n]*\n\s*)?apiSchemas = \[[^\]]*"PhysicsRigidBodyAPI"[^\]]*\]', text))

    # 1) remove every fixed joint anchored to a non-body / missing prim
    removed = []
    for m in re.finditer(r'def PhysicsFixedJoint "(\w+)"', text):
        s, e = find_block(text, m.start())
        block = text[s:e]
        targets = re.findall(r"rel physics:body[01] = <([^>]+)>", block)
        if any(t.rsplit("/", 1)[-1] not in {b.rsplit("/", 1)[-1] for b in bodies} for t in targets):
            text = text[:s] + text[e:]
            removed.append(m.group(1))

    # 2) move the articulation root API from the Geometry Scope to the base
    #    body. The Scope also carries NewtonArticulationRootAPI, which INHERITS
    #    from UsdPhysics.ArticulationRootAPI -- leaving it in place counts as a
    #    second articulation root ("Expected exactly one ArticulationRootAPI").
    text, n = re.subn(
        r'def Scope "Geometry" \(\s*apiSchemas = \[[^\]]*\]\s*\)\s*\{\s*bool newton:selfCollisionEnabled = 0\s*\n',
        'def Scope "Geometry"\n    {\n',
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("could not strip the Geometry Scope root apiSchemas")
    text, n = re.subn(
        r'(def Xform "' + BASE_LINK + r'" \(\s*apiSchemas = \[)"PhysicsRigidBodyAPI"(?=[,\]])',
        r'\1"PhysicsArticulationRootAPI", "PhysicsRigidBodyAPI"',
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError(f"could not add the root API to {BASE_LINK}'s apiSchemas")
    print(f"patched: removed fixed joints {removed}; root API moved onto '{BASE_LINK}'")
    return text


def main() -> int:
    converter = UrdfConverter(CONVERTER_CFG)
    src_path = Path(converter.usd_path)
    print(f"converted: {src_path}")

    patched = patch_usda_text(src_path.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patched_path = BUILD_DIR / "franka_leap_patched.usda"
    patched_path.write_text(patched)

    # export as binary .usd (repo convention: LFS-tracked usd, like the LEAP hand)
    stage = Usd.Stage.Open(str(patched_path))
    if not stage.Export(str(OUT_FILE)):
        raise RuntimeError(f"export failed: {OUT_FILE}")
    print(f"wrote {OUT_FILE}")

    # read-only verification on a pristine stage (no edits -> no listener interference)
    stage = Usd.Stage.Open(str(OUT_FILE))
    root_prims = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    n_rev = sum(1 for p in stage.Traverse() if p.GetTypeName() == "PhysicsRevoluteJoint")
    n_fix = sum(1 for p in stage.Traverse() if p.GetTypeName() == "PhysicsFixedJoint")
    print(f"check: {n_rev} revolute joints, {n_fix} fixed joints, roots={[str(p.GetPath()) for p in root_prims]}")
    assert n_rev == 23, f"expected 23 revolute joints, found {n_rev}"
    assert len(root_prims) == 1 and root_prims[0].GetName() == BASE_LINK, (
        f"expected exactly one articulation root on {BASE_LINK}, found {[str(p.GetPath()) for p in root_prims]}"
    )
    print("OK: single articulation root on the base body (fixed-base encoding)")

    shutil.rmtree(BUILD_DIR)  # converter workspace; regenerated on every run
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
