#!/usr/bin/env python3
"""Shared helpers for building arm + LEAP merged URDFs."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np


def parse_urdf(path: Path) -> tuple[dict, dict]:
    """Minimal URDF <link>/<joint> parser. Returns (links, joints) dicts."""
    root = ET.parse(path).getroot()
    links: dict[str, ET.Element] = {}
    joints: dict[str, ET.Element] = {}
    for elem in root:
        tag = elem.tag.split("}")[-1]
        if tag == "link":
            links[elem.get("name")] = elem
        elif tag == "joint":
            joints[elem.get("name")] = elem
    return links, joints


def urdf_fk(joints: dict, q: dict, start: str, end: str) -> dict[str, np.ndarray]:
    """Forward kinematics of a URDF revolute chain from ``start`` to ``end``."""
    # Parent -> joint map: lets us walk the chain in O(N) instead of O(N^2).
    parent_to_joint: dict[str, str] = {}
    for name, j in joints.items():
        parent = j.find("parent").get("link")
        parent_to_joint[parent] = name

    chain: list[str] = []
    cur = start
    for _ in range(100):
        name = parent_to_joint.get(cur)
        if name is None:
            break
        chain.append(name)
        cur = joints[name].find("child").get("link")
        if cur == end:
            break
    assert cur == end, f"end link {end} not reached, chain={chain}"

    def rot_xyz(rpy: np.ndarray) -> np.ndarray:
        r, p, y = rpy
        cz, sz = np.cos(y), np.sin(y)
        cy, sy = np.cos(p), np.sin(p)
        cx, sx = np.cos(r), np.sin(r)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        return Rz @ Ry @ Rx

    def origin(name: str) -> tuple[np.ndarray, np.ndarray]:
        o = joints[name].find("origin")
        if o is None:
            return np.zeros(3), np.zeros(3)
        xyz = np.array([float(v) for v in o.get("xyz").split()])
        rpy = np.array([float(v) for v in o.get("rpy").split()])
        return xyz, rpy

    T = np.eye(4)
    poses: dict[str, np.ndarray] = {}
    for name in chain:
        xyz, rpy = origin(name)
        T = T @ np.block([[rot_xyz(rpy), xyz[:, None]], [np.zeros((1, 3)), 1]])
        j = joints[name]
        if j.get("type") == "revolute":
            axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            th = q.get(name, 0.0)
            c, s = np.cos(th), np.sin(th)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R = np.eye(3) + s * K + (1 - c) * (K @ K)
            T = T @ np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1]])
        child = j.find("child").get("link")
        poses[child] = T.copy()
    return poses


def write_pretty_urdf(root: ET.Element, path: Path) -> None:
    """Write a URDF root element with pretty XML formatting."""
    xmlstr = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")
    path.write_text("\n".join(line for line in xmlstr.splitlines() if line.strip()) + "\n")


def prepare_leap_urdf(
    links: dict[str, ET.Element],
    joints: dict[str, ET.Element],
    leap_urdf_src: Path,
    assets_urdf: Path,
) -> None:
    """Rename LEAP revolute joints to ``a_N`` and copy meshes into ``assets_urdf``."""
    leap_rev = [n for n, j in joints.items() if j.get("type") == "revolute"]
    if len(leap_rev) != 16:
        raise AssertionError(f"expected 16 leap joints, got {len(leap_rev)}")

    joint_rename = {n: f"a_{n}" for n in leap_rev}
    # Rebuild the dict with new names while preserving document order.
    renamed: dict[str, ET.Element] = {}
    for old_name, elem in joints.items():
        new_name = joint_rename.get(old_name, old_name)
        elem.set("name", new_name)
        renamed[new_name] = elem
    joints.clear()
    joints.update(renamed)

    out_v = assets_urdf / "leap/meshes/visual"
    out_c = assets_urdf / "leap/meshes/collision"
    out_v.mkdir(parents=True, exist_ok=True)
    out_c.mkdir(parents=True, exist_ok=True)
    lvdir = leap_urdf_src / "meshes/visual"
    lcdir = leap_urdf_src / "meshes/collision"

    for link in links.values():
        for g in link.findall(".//mesh"):
            fn = g.get("filename")
            if not fn.startswith("meshes/"):
                raise AssertionError(f"unexpected leap mesh path: {fn}")
            parts = fn.split("/")
            sub, name = parts[1], parts[-1]
            src = (lvdir if sub == "visual" else lcdir) / name
            if not src.exists():
                raise AssertionError(f"missing leap mesh: {src}")
            shutil.copy(src, (out_v if sub == "visual" else out_c) / name)
            g.set("filename", f"leap/meshes/{sub}/{name}")
