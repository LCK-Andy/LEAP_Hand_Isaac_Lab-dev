#!/usr/bin/env python3
"""
Convert downloaded YCB OBJ meshes to physics-enabled USD files.

Parses OBJ vertex/face data and inlines it as UsdGeom.Mesh inside
a USD file with RigidBodyAPI + CollisionAPI applied. No Kit deps.

Usage:
    python scripts/convert_ycb_to_usd.py
    python scripts/convert_ycb_to_usd.py --objects cracker_box,mustard_bottle

Run after: python scripts/download_ycb.py
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
YCB_DIR = REPO_ROOT / "source" / "LEAP_Isaaclab" / "LEAP_Isaaclab" / "assets" / "ycb"


def parse_obj(obj_path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], list[tuple[float, float]], list[tuple[int, int, int]]]:
    """Parse a Wavefront OBJ file. Returns (vertices, faces, uvs, uv_faces)."""
    vertices = []
    faces = []
    uvs = []
    uv_faces = []

    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v':
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == 'vt':
                uvs.append((float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0))
            elif parts[0] == 'f':
                # Face: "f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3" or "f v1//vn1 v2//vn2 v3//vn3"
                face_verts = []
                face_uvs = []
                for v in parts[1:]:
                    indices = v.split('/')
                    face_verts.append(int(indices[0]))
                    if len(indices) > 1 and indices[1]:
                        face_uvs.append(int(indices[1]))

                if len(face_verts) == 3:
                    faces.append((face_verts[0], face_verts[1], face_verts[2]))
                    if face_uvs and len(face_uvs) == 3:
                        uv_faces.append((face_uvs[0], face_uvs[1], face_uvs[2]))
                elif len(face_verts) == 4:
                    # Triangulate quad
                    faces.append((face_verts[0], face_verts[1], face_verts[2]))
                    faces.append((face_verts[0], face_verts[2], face_verts[3]))
                    if face_uvs and len(face_uvs) == 4:
                        uv_faces.append((face_uvs[0], face_uvs[1], face_uvs[2]))
                        uv_faces.append((face_uvs[0], face_uvs[2], face_uvs[3]))

    return vertices, faces, uvs, uv_faces


def create_usd_inline(obj_dir: Path, usd_path: Path) -> bool:
    """Create a USD file with inline mesh data from OBJ."""
    name = obj_dir.name
    friendly = name.split("_", 1)[1] if "_" in name else name

    textured = obj_dir / "textured.obj"
    if not textured.exists():
        print(f"    SKIP: no textured.obj")
        return False

    print("    parsing OBJ...", end=" ", flush=True)
    vertices, faces, uvs, uv_faces = parse_obj(textured)
    print(f"{len(vertices)} verts, {len(faces)} faces")

    # Build faceVertexCounts and faceVertexIndices (USD mesh format)
    face_vertex_counts = [3] * len(faces)  # all triangles
    face_vertex_indices = []
    for f in faces:
        # OBJ faces are 1-indexed
        face_vertex_indices.extend([f[0] - 1, f[1] - 1, f[2] - 1])

    # Compute extent
    if vertices:
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        extent_min = f"({min(xs):.6f}, {min(ys):.6f}, {min(zs):.6f})"
        extent_max = f"({max(xs):.6f}, {max(ys):.6f}, {max(zs):.6f})"
    else:
        extent_min = extent_max = "(-0.5, -0.5, -0.5)"

    # Build UV arrays if available
    uv_section = ""
    if uvs and uv_faces:
        uv_indices = []
        for uf in uv_faces:
            uv_indices.extend([uf[0] - 1, uf[1] - 1, uf[2] - 1])

        st_strs = [f"({u:.6f}, {v:.6f})" for u, v in uvs]
        uv_str = ", ".join(st_strs)

        uv_idx_strs = [str(i) for i in uv_indices]
        uv_idx_str = ", ".join(uv_idx_strs)

        uv_section = f'''
        texCoord2f[] primvars:st = [{uv_str}] (
            interpolation = "vertex"
        )
        int[] primvars:st:indices = [{uv_idx_str}]
'''

    # Build vertex array
    point_strs = [f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})" for v in vertices]
    points_str = ", ".join(point_strs)

    face_counts_str = ", ".join(str(c) for c in face_vertex_counts)
    face_idx_str = ", ".join(str(i) for i in face_vertex_indices)

    usd = f'''#usda 1.0
(
    defaultPrim = "{friendly}"
    upAxis = "Z"
    metersPerUnit = 0.01
    endTimeCode = 0
    startTimeCode = 0
    timeCodesPerSecond = 24
)

def Xform "{friendly}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
)
{{
    uniform bool physics:rigidBodyEnabled = true
    uniform bool physics:kinematicEnabled = false

    def Mesh "mesh" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        int[] faceVertexCounts = [{face_counts_str}]
        int[] faceVertexIndices = [{face_idx_str}]
        point3f[] points = [{points_str}]
        float3[] extent = [{extent_min}, {extent_max}]{uv_section}
        uniform token subdivisionScheme = "none"
    }}
}}
'''
    usd_path.write_text(usd)
    print(f"    wrote {usd_path.stat().st_size:,} bytes")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert YCB OBJ to USD with inline mesh")
    parser.add_argument("--objects", type=str, help="Comma-separated object dir names")
    args = parser.parse_args()

    if not YCB_DIR.exists():
        print(f"YCB directory not found: {YCB_DIR}")
        print("Run: python scripts/download_ycb.py")
        sys.exit(1)

    obj_dirs = []
    for d in sorted(YCB_DIR.iterdir()):
        if not d.is_dir():
            continue
        if (d / "textured.obj").exists():
            if args.objects and d.name not in args.objects.split(","):
                continue
            obj_dirs.append(d)

    if not obj_dirs:
        print("No OBJ directories found. Run download_ycb.py first.")
        sys.exit(1)

    print(f"Converting {len(obj_dirs)} objects to USD with inline mesh...\n")
    created = 0
    failed = []

    for obj_dir in obj_dirs:
        name = obj_dir.name
        friendly = name.split("_", 1)[1] if "_" in name else name
        usd_path = YCB_DIR / f"{friendly}.usd"

        print(f"  [{created + 1}/{len(obj_dirs)}] {friendly}:", end=" ", flush=True)
        if create_usd_inline(obj_dir, usd_path):
            created += 1
        else:
            failed.append(friendly)

    print(f"\nCreated: {created}, failed: {len(failed)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()