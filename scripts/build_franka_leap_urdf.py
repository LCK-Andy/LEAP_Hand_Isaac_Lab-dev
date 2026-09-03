#!/usr/bin/env python3
"""Build franka_leap.urdf — a LEAP hand welded onto a Franka Panda flange.

Creates a single 23-DOF articulation (Franka 7 + LEAP 16) by merging:

  * Franka Panda arm (7-DOF, up to the panda_link8 flange) from the Isaac Gym
    franka_description URDF
  * LEAP hand (16-DOF) from the DRO-Grasp leap_hand_right.urdf (official LEAP
    geometry, joints renamed a_0..a_15 to match the LEAP_Hand_Isaac_Lab
    project convention)

The weld fixed joint maps the LEAP wrist frame onto the flange such that the
palm normal (LEAP base +X) aligns with the flange tool axis (+Z): with the
arm in its default ready pose the palm faces down and the fingers point
forward (flange +X). Meshes are copied into the project assets so the
resulting URDF is fully self-contained (offline conversion).

Run: python scripts/build_franka_leap_urdf.py
"""

from __future__ import annotations

import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Allow importing shared helpers from the scripts directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from urdf_utils import parse_urdf, prepare_leap_urdf, urdf_fk, write_pretty_urdf

# ---------------------------------------------------------------- paths
PROJECT = Path(__file__).resolve().parents[1]
ASSETS_URDF = PROJECT / "source/LEAP_Isaaclab/LEAP_Isaaclab/assets/urdf"
FRANKA_URDF_SRC = Path(
    os.environ.get("FRANKA_URDF_SRC", "/home/andy/Documents/DexGraspNet2/isaacgym/assets/urdf")
)
LEAP_URDF_SRC = Path(
    os.environ.get("LEAP_URDF_SRC", "/home/andy/Documents/DRO-Grasp/data/data_urdf/robot/leaphand")
)

FRANKA_URDF = FRANKA_URDF_SRC / "franka_description/robots/franka_panda.urdf"
LEAP_URDF = LEAP_URDF_SRC / "leap_hand_right.urdf"
OUT_URDF = ASSETS_URDF / "franka_leap.urdf"

# Weld transform: panda_link8 -> LEAP base frame.
# R = Rz(-90 deg) * Ry(-90 deg): base +X (palm normal) -> flange +Z,
# base +Y (fingers) -> flange +X, right-handed.
WELD_XYZ = "0 0 0"
WELD_RPY = "0 -1.5707963267948966 -1.5707963267948966"

# Franka default joint positions (same as isaaclab_assets FRANKA_PANDA_CFG)
FRANKA_DEFAULT_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
}

# Joints/links dropped from the Franka URDF (panda hand + gripper);
# the arm is kept up to the flange (panda_link8, via fixed panda_joint8).
FRANKA_DROP_LINKS = {
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
}
FRANKA_DROP_JOINTS = {
    "panda_hand_joint",
    "panda_finger_joint1",
    "panda_finger_joint2",
}


def rewrite_franka_meshes(
    links: dict[str, ET.Element],
    src_visual_dir: Path,
    src_collision_dir: Path,
    out_visual_dir: Path,
    out_collision_dir: Path,
) -> None:
    """Rewrite package://franka_description mesh paths and copy meshes into assets."""
    out_visual_dir.mkdir(parents=True, exist_ok=True)
    out_collision_dir.mkdir(parents=True, exist_ok=True)

    for link in links.values():
        for g in link.findall(".//mesh"):
            fn = g.get("filename")
            if not fn.startswith("package://franka_description/"):
                continue
            rel = fn.replace("package://franka_description/", "")
            parts = rel.split("/")
            sub = parts[1]  # "visual" or "collision"
            name = parts[-1]
            src_dir = src_visual_dir if sub == "visual" else src_collision_dir
            src = src_dir / name
            obj = src.with_suffix(".obj")
            src_file = obj if obj.exists() else src
            if sub == "visual" and src_file.suffix != ".obj":
                print(f"  [warn] no obj for franka visual {name}, using {src_file.suffix}")
            target_dir = out_visual_dir if sub == "visual" else out_collision_dir
            shutil.copy(src_file, target_dir / src_file.name)
            g.set("filename", f"franka/meshes/{sub}/{src_file.name}")


def main() -> int:
    links_f, joints_f = parse_urdf(FRANKA_URDF)
    links_l, joints_l = parse_urdf(LEAP_URDF)

    # ---- checks -------------------------------------------------------
    missing = FRANKA_DROP_LINKS - set(links_f)
    assert not missing, f"franka links missing: {missing}"
    assert {"panda_link0", "panda_link7"} <= set(links_f), "franka chain incomplete"
    assert {"panda_joint1", "panda_joint7"} <= set(joints_f)
    leap_rev = [n for n, j in joints_l.items() if j.get("type") == "revolute"]
    assert len(leap_rev) == 16, f"expected 16 leap joints, got {len(leap_rev)}"
    assert "base" in links_l and "palm_lower" in links_l

    # ---- franka: strip gripper, rewrite mesh paths ---------------------
    for j in FRANKA_DROP_JOINTS:
        joints_f.pop(j)
    for l in FRANKA_DROP_LINKS:
        links_f.pop(l)

    # The source URDF has panda_joint8 (flange) commented out; re-add the
    # standard Franka flange: fixed joint panda_link7 -> panda_link8 at z=0.107
    links_f["panda_link8"] = ET.Element("link", {"name": "panda_link8"})
    fj = ET.Element("joint", {"name": "panda_joint8", "type": "fixed"})
    ET.SubElement(fj, "origin", {"xyz": "0 0 0.107", "rpy": "0 0 0"})
    ET.SubElement(fj, "parent", {"link": "panda_link7"})
    ET.SubElement(fj, "child", {"link": "panda_link8"})
    joints_f["panda_joint8"] = fj

    rewrite_franka_meshes(
        links_f,
        src_visual_dir=FRANKA_URDF_SRC / "franka_description/meshes/visual",
        src_collision_dir=FRANKA_URDF_SRC / "franka_description/meshes/collision",
        out_visual_dir=ASSETS_URDF / "franka/meshes/visual",
        out_collision_dir=ASSETS_URDF / "franka/meshes/collision",
    )

    # ---- leap: rename joints to a_N, fix mesh paths ---------------------
    prepare_leap_urdf(links_l, joints_l, LEAP_URDF_SRC, ASSETS_URDF)

    # ---- merged robot --------------------------------------------------
    root = ET.Element("robot", {"name": "franka_leap"})

    for name in [f"panda_link{i}" for i in range(9)]:
        root.append(links_f[name])
    for name in [f"panda_joint{i}" for i in range(1, 9)]:
        root.append(joints_f[name])

    weld = ET.SubElement(root, "joint", {"name": "leap_weld_joint", "type": "fixed"})
    ET.SubElement(weld, "origin", {"xyz": WELD_XYZ, "rpy": WELD_RPY})
    ET.SubElement(weld, "parent", {"link": "panda_link8"})
    ET.SubElement(weld, "child", {"link": "base"})

    for elem in links_l.values():
        root.append(elem)
    for elem in joints_l.values():
        root.append(elem)

    write_pretty_urdf(root, OUT_URDF)
    print(f"wrote {OUT_URDF}")

    # ---- FK sanity check: flange orientation at ready pose --------------
    import numpy as np

    poses = urdf_fk(joints_f, FRANKA_DEFAULT_JOINT_POS, "panda_link0", "panda_link8")
    T = poses["panda_link8"]
    flange_pos = T[:3, 3]
    flange_z = T[:3, 2]

    r, p, y = [float(v) for v in WELD_RPY.split()]
    cz, sz = np.cos(y), np.sin(y)
    cy, sy = np.cos(p), np.sin(p)
    cx, sx = np.cos(r), np.sin(r)
    R_w = (
        np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    )
    palm_normal_world = T[:3, :3] @ R_w[:, 0]
    fingers_world = T[:3, :3] @ R_w[:, 1]

    print(f"flange pos (default pose): {np.round(flange_pos, 4)}")
    print(f"flange z (tool) axis     : {np.round(flange_z, 4)}")
    print(f"palm normal (base +X)    : {np.round(palm_normal_world, 4)}")
    print(f"fingers (base +Y)        : {np.round(fingers_world, 4)}")
    assert np.linalg.norm(flange_z) > 0.99, "FK error"
    assert np.allclose(palm_normal_world, flange_z, atol=0.05), "palm normal not aligned with flange tool axis"
    print("FK check OK: palm normal is aligned with the flange tool axis.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
