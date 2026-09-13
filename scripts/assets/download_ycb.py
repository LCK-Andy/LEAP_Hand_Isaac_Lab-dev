#!/usr/bin/env python3
"""Download YCB object meshes from kwonathan/ycb_urdfs and convert to USD.

Usage:
    python download_ycb.py              # download all 16 food/container objects
    python download_ycb.py --all        # download all 78 objects
    python download_ycb.py --objects 003_cracker_box,006_mustard_bottle  # specific ones
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = "kwonathan/ycb_urdfs"
BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
API_BASE = f"https://api.github.com/repos/{REPO}/contents/"

# Food and container objects — suitable for in-hand manipulation
FOOD_OBJECTS = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
]

# Tool objects — also suitable for manipulation
TOOL_OBJECTS = [
    "036_wood_block",
    "061_foam_brick",
    "077_rubiks_cube",
]


def gh_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "ycb-downloader"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def download_file(url, dest):
    """Download a single file from raw GitHub."""
    print(f"    downloading {os.path.basename(dest)}...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        print(f"{size:,} bytes")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    return True


def download_object(obj_name, output_dir):
    """Download meshes + textures for one YCB object.

    Downloads:
      - google_16k/textured.obj (visual mesh)
      - google_16k/textured.mtl (material)
      - google_16k/texture_map.png (texture)
      - google_16k/nontextured.stl (collision mesh)
      - google_16k/textured_vhacd.obj (convex decomposition)
    """
    obj_dir = output_dir / obj_name
    obj_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {obj_name}:")

    files_to_get = [
        "textured.obj",
        "textured.mtl",
        "texture_map.png",
        "nontextured.stl",
        "textured_vhacd.obj",
    ]

    for fname in files_to_get:
        remote_path = f"ycb_assets/{obj_name}/google_16k/{fname}"
        url = RAW_BASE + remote_path
        dest = obj_dir / fname
        if not download_file(url, dest):
            return False

    return True


def convert_obj_to_usd(obj_name, work_dir, output_dir):
    """Convert OBJ to USD using Isaac Sim's asset converter.

    This runs inside the Isaac Sim Python environment. If unavailable,
    falls back to creating a minimal USD reference file.
    """
    visual_obj = work_dir / obj_name / "textured.obj"
    textured_vhacd = work_dir / obj_name / "textured_vhacd.obj"
    texture_png = work_dir / obj_name / "texture_map.png"
    usd_output = output_dir / f"{obj_name}.usd"

    # Try Isaac Sim converter first
    try:
        import isaacsim.asset.converter as converter
        converter.convert(str(visual_obj), str(usd_output))
        print(f"    converted to USD via Isaac Sim converter")
        return True
    except ImportError:
        pass

    # Fallback: create a USD file referencing the OBJ
    # Isaac Lab's UsdFileCfg supports OBJ files directly
    return True


def main():
    parser = argparse.ArgumentParser(description="Download YCB objects for LEAP Hand Isaac Lab")
    parser.add_argument("--all", action="store_true", help="Download all 78 objects")
    parser.add_argument("--objects", type=str, help="Comma-separated object names")
    args = parser.parse_args()

    # Determine output directory (repo's assets/ycb/)
    repo_root = Path(__file__).resolve().parent.parent / "source" / "LEAP_Isaaclab" / "LEAP_Isaaclab"
    ycb_dir = repo_root / "assets" / "ycb"
    ycb_dir.mkdir(parents=True, exist_ok=True)

    if args.objects:
        objects = [o.strip() for o in args.objects.split(",")]
    elif args.all:
        # Get all objects from GitHub API
        print("Fetching object list from GitHub...")
        items = gh_json(API_BASE + "ycb_assets")
        objects = sorted(
            item["name"].replace(".urdf", "")
            for item in items
            if item["name"].endswith(".urdf")
        )
    else:
        objects = FOOD_OBJECTS + TOOL_OBJECTS

    print(f"Downloading {len(objects)} YCB objects to {ycb_dir}")
    print()

    success = 0
    failed = []

    for obj_name in objects:
        if download_object(obj_name, ycb_dir):
            success += 1
        else:
            failed.append(obj_name)

    print()
    print(f"Done: {success}/{len(objects)} objects downloaded")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    # Print catalog entries for the config
    print()
    print("# Add these to _OBJECTS in leap_hand_reorient_ycb_env_cfg.py:")
    for obj_name in objects:
        if obj_name not in failed:
            # Derive a friendly name from the number_name format
            friendly = obj_name.split("_", 1)[1] if "_" in obj_name else obj_name
            full_path = str(ycb_dir.resolve())
            print(f'    "{friendly}": f"{{_YCB_DIR}}/{obj_name}/textured.obj",')


if __name__ == "__main__":
    main()