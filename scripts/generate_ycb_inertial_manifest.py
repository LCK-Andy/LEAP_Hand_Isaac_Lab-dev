#!/usr/bin/env python
"""Generate the YCB inertial manifest consumed by Isaac-Repose-Objects-LEAP-Direct-v1.

Reads the sibling repo's ground-truth density dataset (Estimate_weight_distribution:
real YCB objects x random density strategies, tet-mesh point-mass ground truth) and
bakes a standalone JSON holding the FULL inertial state per (object, strategy):
mass [kg], COM [m], inertia [kg*m^2] about the COM, principal moments/axes plus
their quaternion, and the per-entry friction pair. The v1 env then needs no runtime
dependency on the sibling repo.

Frames/units: values live in the raw YCB canonical mesh frame (meters) -- the same
frame as this repo's wrapper USDs (identity root transform, metersPerUnit = 1), so
they transfer unmodified into the USD body-local frame.

The sibling's ``synthetic_density/inertial.py`` is imported STANDALONE (importlib
from file path): the package __init__ pulls ``gmsh`` (heavy, generation-only),
while loading only needs networkx/defusedxml/numpy.

Usage (inside env_isaaclab; pip-installed defusedxml is required):
  ~/env_isaaclab/bin/python scripts/generate_ycb_inertial_manifest.py \
      [--repo ~/Documents/Estimate_weight_distribution] [--objects strawberry,lemon,...]
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/assets/ycb/inertial_manifest.json"
# the synthetic cube/sphere of the sibling repo are ~0.5 m / hundreds of kg --
# nothing a LEAP palm holds; real YCB objects only by default
SYNTHETIC_OBJECTS = {"cube", "sphere"}


def load_sibling_inertial_module(repo: Path):
    """Import the sibling repo's inertial.py as a standalone module."""
    path = repo / "synthetic_density" / "inertial.py"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (pass --repo?)")
    spec = importlib.util.spec_from_file_location("sibling_inertial", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sibling_inertial"] = module  # dataclass processing needs the module registered
    spec.loader.exec_module(module)
    return module


def quat_from_matrix(r: np.ndarray) -> list[float]:
    """Unit quaternion (w, x, y, z) of rotation matrix ``r`` (Shepperd's method)."""
    t = float(np.trace(r))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def quat_to_matrix(q: list[float]) -> np.ndarray:
    """Rotation matrix of quaternion ``q`` = (w, x, y, z) -- for round-trip checks."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path("~/Documents/Estimate_weight_distribution").expanduser())
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--objects",
        default="",
        help="comma-separated object names to include (default: every real YCB object in the "
        f"sibling manifest, i.e. all except the synthetic {sorted(SYNTHETIC_OBJECTS)})",
    )
    args = parser.parse_args()

    inertial = load_sibling_inertial_module(args.repo)
    with open(args.repo / "data/density/manifest.json") as f:
        sibling_manifest = json.load(f)

    if args.objects:
        wanted = {name.strip() for name in args.objects.split(",") if name.strip()}
    else:
        wanted = {e["object"] for e in sibling_manifest} - SYNTHETIC_OBJECTS

    entries = [e for e in sibling_manifest if e["object"] in wanted]
    missing = wanted - {e["object"] for e in entries}
    if missing:
        raise KeyError(f"objects absent from the sibling manifest: {sorted(missing)}")

    out_objects: dict[str, list[dict]] = {}
    for entry in sorted(entries, key=lambda e: (e["object"], e["strategy"])):
        props = inertial.load_inertial_properties(args.repo / "data/density" / entry["file"])

        # gate on the sibling repo's own physical-plausibility checks ...
        problems = props.validate()
        if problems:
            raise ValueError(f"{entry['file']}: {problems}")
        # ... and on manifest <-> recomputation consistency (mass, COM, moments)
        def rel(a, b, scale):
            return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / max(scale, 1e-30)

        bbox_diag = float(np.linalg.norm(props.bbox_max - props.bbox_min))
        errs = (
            rel(props.total_mass, entry["total_mass"], entry["total_mass"]),
            rel(props.com, entry["com"], bbox_diag),
            rel(props.principal_moments, entry["principal_moments"], entry["principal_moments"][0]),
        )
        if max(errs) > 1e-6:
            raise ValueError(f"{entry['file']}: manifest mismatch, rel errors {errs}")

        axes = props.principal_axes.copy()
        if np.linalg.det(axes) < 0:  # eigh may return a left-handed frame; inertia needs R diag R^T
            axes[:, 2] *= -1.0
        quat = quat_from_matrix(axes)
        if not np.allclose(quat_to_matrix(quat) @ np.diag(props.principal_moments) @ quat_to_matrix(quat).T,
                           props.inertia, atol=1e-9 * max(1.0, entry["principal_moments"][0])):
            raise AssertionError(f"{entry['file']}: quaternion round-trip failed")

        out_objects.setdefault(entry["object"], []).append(
            {
                "strategy": entry["strategy"],
                "repeat": entry.get("repeat", 0),
                "mass": float(props.total_mass),
                "com": props.com.tolist(),
                # full 3x3 about the COM, body-local == YCB canonical mesh frame [kg*m^2]
                "inertia": props.inertia.tolist(),
                "principal_moments": props.principal_moments.tolist(),  # descending
                "principal_axes": axes.tolist(),  # columns are the axes
                "principal_axes_quat_wxyz": quat,
                "mu_static": float(entry.get("mu_static", float("nan"))),
                "mu_dynamic": float(entry.get("mu_dynamic", float("nan"))),
                "total_volume": float(props.total_volume),
                "density_mean": float(props.density_mean),
                "source_file": entry["file"],
            }
        )
        print(
            f"  {entry['object']:12s} {entry['strategy']:8s} m={props.total_mass:7.3f} kg  "
            f"com={np.round(props.com, 4).tolist()}  I_diag={np.round(props.principal_moments, 6).tolist()}"
        )

    output = {
        "_doc": (
            "Ground-truth YCB inertial variants for Isaac-Repose-Objects-LEAP-Direct-v1. "
            "Generated by scripts/generate_ycb_inertial_manifest.py from the sibling "
            "Estimate_weight_distribution density dataset (tet-mesh point-mass ground truth). "
            "Units: kg / m / kg*m^2. Frames: YCB canonical mesh frame == the wrapper-USD "
            "body-local frame. Inertia is about the COM; inertia = principal_axes @ "
            "diag(principal_moments) @ principal_axes.T; principal_axes_quat_wxyz is the "
            "quaternion of principal_axes."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_repo": str(args.repo),
        "objects": out_objects,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=1)
    n = sum(len(v) for v in out_objects.values())
    print(f"\nwrote {n} variants over {len(out_objects)} objects -> {args.out}")


if __name__ == "__main__":
    main()
