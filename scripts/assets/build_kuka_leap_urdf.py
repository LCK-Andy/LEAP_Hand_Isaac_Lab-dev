#!/usr/bin/env python3
"""Build kuka_leap.urdf — a LEAP hand welded onto a Kuka iiwa7 flange.

Creates a single 23-DOF articulation (Kuka iiwa7 7-DOF + LEAP 16-DOF) by merging:

  * Kuka iiwa7 arm (7-DOF, up to the iiwa7_link_7 wrist link) from the Isaac Gym
    kuka_allegro_description URDF (official Kuka geometry)
  * LEAP hand (16-DOF) from the DRO-Grasp leap_hand_right.urdf (official LEAP
    geometry, joints renamed a_0..a_15 to match the LEAP_Hand_Isaac_Lab
    project convention)

The weld fixed joint maps the LEAP wrist frame onto the flange such that the
palm normal (LEAP base +X) aligns with the flange tool axis (+Z): with the
arm in its default ready pose the palm faces down-ish and the fingers point
forward, like a wrist extension on the Kuka-Allegro mount. Meshes are copied
into the project assets so the resulting URDF is fully self-contained
(offline conversion).

Run: python scripts/build_kuka_leap_urdf.py
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
KUKA_URDF_SRC = Path(
    os.environ.get(
        "KUKA_URDF_SRC",
        "/home/andy/Documents/DexGraspNet2/isaacgym/assets/urdf/kuka_allegro_description",
    )
)
LEAP_URDF_SRC = Path(
    os.environ.get("LEAP_URDF_SRC", "/home/andy/Documents/DRO-Grasp/data/data_urdf/robot/leaphand")
)

KUKA_URDF = KUKA_URDF_SRC / "kuka.urdf"
LEAP_URDF = LEAP_URDF_SRC / "leap_hand_right.urdf"
OUT_URDF = ASSETS_URDF / "kuka_leap.urdf"

# Weld transform: iiwa7_link_7 -> LEAP base frame.
# R = Rz(-90 deg) * Ry(0) * Rx(+105.255 deg): base +X (palm normal) -> world -Z
# (palm down at the Kuka-Allegro ready pose), base +Y (fingers) -> world
# (-0.496, 0.868, 0) i.e. horizontal, angled toward the object. This matches
# the arm/object geometry of the original Dexsuite Kuka-Allegro env so the
# hand approaches the object palm-down like the Allegro mount does.
# xyz includes the iiwa7_link_7 -> flange offset (0 0 0.071) equivalent to the
# original iiwa7_joint_ee fixed joint, so iiwa7_link_ee is dropped from the
# merged robot.
WELD_XYZ = "0 0 0.05"
WELD_RPY = "0.0 0.0 -1.57079633"

# Kuka iiwa7 default joint positions (same as isaaclab_assets KUKA_ALLEGRO_CFG)
KUKA_DEFAULT_JOINT_POS = {
    "iiwa7_joint_1": 0.0,
    "iiwa7_joint_2": 0.0,
    "iiwa7_joint_3": 0.7854,
    "iiwa7_joint_4": 1.5708,
    "iiwa7_joint_5": -1.5708,
    "iiwa7_joint_6": -1.5708,
    "iiwa7_joint_7": 0.0,
}

# Joints/links kept in chain order: base fixed link, 0..7, flange dropped.
KUKA_LINK_ORDER = ["iiwa7_base_link", "iiwa7_link_0", "iiwa7_link_1", "iiwa7_link_2",
                   "iiwa7_link_3", "iiwa7_link_4", "iiwa7_link_5", "iiwa7_link_6", "iiwa7_link_7"]
KUKA_JOINT_ORDER = ["iiwa7_base_link_iiwa7_joint"] + [f"iiwa7_joint_{i}" for i in range(1, 8)]
START_LINK = "iiwa7_base_link"
END_LINK = "iiwa7_link_7"


def main() -> int:
    links_k, joints_k = parse_urdf(KUKA_URDF)
    links_l, joints_l = parse_urdf(LEAP_URDF)

    # ---- checks -------------------------------------------------------
    assert set(KUKA_LINK_ORDER) <= set(links_k), f"kuka links missing: {set(KUKA_LINK_ORDER) - set(links_k)}"
    assert set(KUKA_JOINT_ORDER) <= set(joints_k), f"kuka joints missing: {set(KUKA_JOINT_ORDER) - set(joints_k)}"
    leap_rev = [n for n, j in joints_l.items() if j.get("type") == "revolute"]
    assert len(leap_rev) == 16, f"expected 16 leap joints, got {len(leap_rev)}"
    assert "base" in links_l and "palm_lower" in links_l

    # ---- kuka: rewrite mesh paths --------------------------------------
    out_kv = ASSETS_URDF / "kuka/meshes/visual"
    out_kc = ASSETS_URDF / "kuka/meshes/collision"
    out_kv.mkdir(parents=True, exist_ok=True)
    out_kc.mkdir(parents=True, exist_ok=True)
    kvdir = KUKA_URDF_SRC / "meshes/iiwa7/visual"
    kcdir = KUKA_URDF_SRC / "meshes/iiwa7/collision"

    for link in links_k.values():
        for g in link.findall(".//mesh"):
            fn = g.get("filename")
            assert fn.startswith("kuka_allegro_description/"), f"unexpected kuka mesh path: {fn}"
            # kuka_allegro_description/meshes/iiwa7/{visual,collision}/link_N.obj
            parts = fn.split("/")
            sub, name = parts[3], parts[-1]
            src = (kvdir if sub == "visual" else kcdir) / name
            assert src.exists(), f"missing kuka mesh: {src}"
            shutil.copy(src, (out_kv if sub == "visual" else out_kc) / name)
            g.set("filename", f"kuka/meshes/{sub}/{name}")

    # ---- leap: rename joints to a_N, fix mesh paths ---------------------
    prepare_leap_urdf(links_l, joints_l, LEAP_URDF_SRC, ASSETS_URDF)

    # ---- merged robot --------------------------------------------------
    root = ET.Element("robot", {"name": "kuka_leap"})

    # kuka links/joints in chain order
    for name in KUKA_LINK_ORDER:
        root.append(links_k[name])
    for name in KUKA_JOINT_ORDER:
        root.append(joints_k[name])

    weld = ET.SubElement(root, "joint", {"name": "leap_weld_joint", "type": "fixed"})
    ET.SubElement(weld, "origin", {"xyz": WELD_XYZ, "rpy": WELD_RPY})
    ET.SubElement(weld, "parent", {"link": "iiwa7_link_7"})
    ET.SubElement(weld, "child", {"link": "base"})

    # leap: keep original document order of links/joints from source urdf
    for elem in links_l.values():
        root.append(elem)
    for elem in joints_l.values():
        root.append(elem)

    write_pretty_urdf(root, OUT_URDF)
    print(f"wrote {OUT_URDF}")

    # ---- FK sanity check: hand orientation at ready pose --------------
    import numpy as np

    poses = urdf_fk(joints_k, KUKA_DEFAULT_JOINT_POS, START_LINK, END_LINK)
    T = poses[END_LINK]
    flange_z = T[:3, 2]
    print(f"flange z (tool) axis     : {np.round(flange_z, 4)}")
    # apply weld rotation: R = Rz(y) Ry(p) Rx(r)
    r, p, y = [float(v) for v in WELD_RPY.split()]
    Rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = lambda a: np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = lambda a: np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    R_w = Rz(y) @ Ry(p) @ Rx(r)
    palm_normal_world = T[:3, :3] @ R_w[:, 0]
    fingers_world = T[:3, :3] @ R_w[:, 1]
    print(f"palm normal (base +X)   : {np.round(palm_normal_world, 4)}")
    print(f"fingers (base +Y)       : {np.round(fingers_world, 4)}")
    print("  (target: palm normal ~= -Z world, fingers horizontal)")
    assert np.linalg.norm(flange_z) > 0.99, "FK error"
    assert abs(np.dot(palm_normal_world, np.array([0.0, 0.0, 1.0])) + 1.0) < 0.05, "palm not down"
    assert abs(fingers_world[2]) < 0.1, "fingers not horizontal"
    print("FK check OK: palm normal is aligned with -Z (palm down) at ready pose.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
