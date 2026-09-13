#!/usr/bin/env python3
"""
Generate sphere-proxy collision USDs for the YCB catalog (MorphIt outputs).

Takes the MorphIt sphere sets (~/Documents/projects/MorphIt/results/ycb/<name>.json,
produced by MorphIt/run_ycb.py with the MorphIt-S surface preset) and derives a new
USD per object from the existing wrapper USD:
  * the visual mesh keeps rendering, but its triangle-mesh collision is DISABLED
    (no more PhysX "falling back to convexHull" errors);
  * K UsdGeom.Sphere prims with PhysicsCollisionAPI provide the collision proxy.

Run (uses plain pxr, no Kit):
  ~/env_isaaclab/bin/python scripts/generate_sphere_proxy_usd.py
  ~/env_isaaclab/bin/python scripts/generate_sphere_proxy_usd.py \
      --results ~/Documents/projects/MorphIt/results/ycb50 --out <YCB_DIR>/spheres50
"""

import argparse
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

REPO = Path(__file__).parent.parent
YCB_DIR = REPO / "source/LEAP_Isaaclab/LEAP_Isaaclab/assets/ycb"
MORPHIT_RESULTS = Path.home() / "Documents/projects/MorphIt/results/ycb"
OUT_DIR = YCB_DIR / "spheres"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=MORPHIT_RESULTS, help="MorphIt results dir")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output USD dir")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for result_path in sorted(args.results.glob("*.json")):
        name = result_path.stem
        wrapper_path = YCB_DIR / f"{name}.usd"
        if not wrapper_path.exists():
            print(f"[spheres] skip {name}: no wrapper USD at {wrapper_path}")
            continue
        data = json.loads(result_path.read_text())
        centers, radii = data["centers"], data["radii"]

        stage = Usd.Stage.Open(str(wrapper_path))
        root = stage.GetDefaultPrim()
        if not root.IsValid():
            print(f"[spheres] skip {name}: wrapper has no default prim")
            continue
        root_path = root.GetPath()

        # visual mesh stays; strip its collision API entirely (disabling it via
        # physics:collisionEnabled still makes PhysX parse-and-warn on the triangle mesh)
        mesh = stage.GetPrimAtPath(root_path.AppendChild("mesh"))
        if mesh.IsValid() and mesh.HasAPI(UsdPhysics.CollisionAPI):
            mesh.RemoveAPI(UsdPhysics.CollisionAPI)
        # the wrapper root also carries an (empty) CollisionAPI; spheres own collision now
        if root.HasAPI(UsdPhysics.CollisionAPI):
            root.RemoveAPI(UsdPhysics.CollisionAPI)

        # sphere collision proxy (centers/radii are in the mesh's local frame, meters)
        for idx, (center, radius) in enumerate(zip(centers, radii)):
            sphere = UsdGeom.Sphere.Define(stage, root_path.AppendChild(f"sphere_{idx}"))
            sphere.CreateRadiusAttr(float(radius))
            xform = UsdGeom.Xformable(sphere)
            xform.AddTranslateOp().Set(Gf.Vec3d(*center))
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())

        out_path = args.out / f"{name}_spheres.usd"
        stage.GetRootLayer().Export(str(out_path))
        print(f"[spheres] {name}: {len(radii)} spheres -> {out_path}")


if __name__ == "__main__":
    main()
