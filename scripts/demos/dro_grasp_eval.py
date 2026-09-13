"""D(R,O) LEAP grasp evaluation in Isaac Lab — CORA / penetration / 6-direction hold success.

Success protocol (paper-faithful): object disturbance = 0.5 * mass along each of 6 axes
(zero-g equivalent of the paper's force test) — implemented as gravity vector g = 0.5*dir
since the external-force API is broken in this Isaac Lab build. Success = object
displacement < 2 cm over 1 s for ALL 6 directions.

Usage (env_isaaclab, from LEAP_Hand_Isaac_Lab repo root):
    python scripts/dro_grasp_eval.py --objects ycb_potted_meat_can
    python scripts/dro_grasp_eval.py   # all 10 objects
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--q_dir", type=str, default="/home/andy/Documents/DRO-Grasp/tmp/isaac_eval")
parser.add_argument("--objects", type=str, default="", help="comma-separated slugs; empty = all")
parser.add_argument("--out_dir", type=str,
                    default="/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab/logs/dro_eval")
parser.add_argument("--render", action="store_true")
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from LEAP_Isaaclab.assets.leap import LEAP_HAND_CFG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = f"{ROOT}/source/LEAP_Isaaclab/LEAP_Isaaclab/assets"
DRO = "/home/andy/Documents/DRO-Grasp"
LINK_PCS_NPZ = f"{ROOT}/assets_eval/leap_links_pc.npz"
DRO_FK_ZERO_NPZ = "/tmp/dro_fk_frames.npz"

DRO_JOINT_ORDER = ['1', '0', '2', '3', '5', '4', '6', '7', '9', '8', '10', '11', '12', '13', '14', '15']
DRO_LINK_TO_BODY = {'palm_lower': 'base'}

OBJECTS = {
    'contactdb_apple': ('contactdb/apple.usd', 'contactdb/apple/apple.stl'),
    'contactdb_camera': ('contactdb/camera.usd', 'contactdb/camera/camera.stl'),
    'contactdb_cylinder_medium': ('contactdb/cylinder_medium.usd', 'contactdb/cylinder_medium/cylinder_medium.stl'),
    'contactdb_door_knob': ('contactdb/door_knob.usd', 'contactdb/door_knob/door_knob.stl'),
    'contactdb_rubber_duck': ('contactdb/rubber_duck.usd', 'contactdb/rubber_duck/rubber_duck.stl'),
    'contactdb_water_bottle': ('contactdb/water_bottle.usd', 'contactdb/water_bottle/water_bottle.stl'),
    'ycb_baseball': ('ycb/baseball.usd', 'ycb/baseball/baseball.stl'),
    'ycb_pear': ('ycb/pear.usd', 'ycb/pear/pear.stl'),
    'ycb_potted_meat_can': ('ycb/potted_meat_can.usd', 'ycb/potted_meat_can/potted_meat_can.stl'),
    'ycb_tomato_soup_can': ('ycb/tomato_soup_can.usd', 'ycb/tomato_soup_can/tomato_soup_can.stl'),
}
DIRECTIONS = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)]


def euler_xyz_to_quat_xyzw(e):
    return Rotation.from_euler("XYZ", e).as_quat()


def quat_to_mat_xyzw(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-1)


def get_mass(slug):
    ds, name = slug.split('_', 1)
    urdf = f"{DRO}/data/data_urdf/object/{ds}/{name}/{name}.urdf"
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(urdf)
        for mass_el in tree.iter('mass'):
            v = mass_el.get('value')
            if v:
                return float(v)
    except Exception:
        pass
    return 0.1


@configclass
class EvalSceneCfg(InteractiveSceneCfg):
    env_spacing = 2.0
    hand: ArticulationCfg = LEAP_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/leap_hand",
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=50.0,
                velocity_limit_sim=100.0,
                stiffness=20.0,
                damping=1.0,
                friction=0.01,
            ),
        },
    )
    obj: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/obj",
        spawn=sim_utils.UsdFileCfg(usd_path=ASSETS + "/ycb/potted_meat_can.usd"),
    )


def run_direction(slug, g_vec, do_metrics):
    """Build a scene, spawn closed hand, settle, (metrics), hold test. Returns per-env results."""
    usd_rel, stl_rel = OBJECTS[slug]
    usd_path = f"{ASSETS}/{usd_rel}"
    stl_path = f"{DRO}/data/data_urdf/object/{stl_rel}"
    mesh = trimesh.load_mesh(stl_path)

    data = np.load(f"{args_cli.q_dir}/{slug}.npz")
    predict_q = data["predict_q"]
    fk_pc = data["fk_pc"]
    n = predict_q.shape[0]

    try:
        sim_utils.SimulationContext.clear_instance()
    except Exception:
        pass
    scene_cfg = EvalSceneCfg(num_envs=n)
    scene_cfg.obj.spawn.usd_path = usd_path
    mass = get_mass(slug)
    scene_cfg.obj.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=False, disable_gravity=False, max_depenetration_velocity=1000.0)
    scene_cfg.obj.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=mass)
    sim_cfg = SimulationCfg(dt=1.0 / 120.0, gravity=tuple(g_vec), device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    hand: Articulation = scene["hand"]
    obj: RigidObject = scene["obj"]
    joint_names = hand.joint_names
    body_names = hand.body_names
    if do_metrics:
        print(f"[{slug}] joints={len(joint_names)} bodies={len(body_names)} grasps={n}")

    dof_map = [joint_names.index(f"a_{m}") for m in DRO_JOINT_ORDER]
    link_order = [str(x) for x in data["fk_link_order"]]
    body_map = {}
    for link in link_order:
        bname = DRO_LINK_TO_BODY.get(link, link)
        assert bname in body_names, f"body {bname} not found: {body_names}"
        body_map[link] = body_names.index(bname)

    # T_corr / T_palm2base (fixed rigid transform between DRO palm frame and USD base frame)
    palm_zero = np.load(DRO_FK_ZERO_NPZ)
    links_z = [str(x) for x in palm_zero["links"]]
    T_palm0 = palm_zero["zero_frames"][links_z.index("palm_lower")]
    zpose = torch.zeros(n, 7, dtype=torch.float32, device=sim.device)
    zpose[:, 6] = 1.0
    hand.write_root_pose_to_sim(zpose)
    qz = torch.zeros(n, hand.num_joints, device=sim.device)
    hand.write_joint_state_to_sim(qz, torch.zeros_like(qz))
    for _ in range(30):
        sim.step()
    qb = hand.data.body_quat_w[0, body_names.index("base")].cpu().numpy()
    pb = hand.data.body_pos_w[0, body_names.index("base")].cpu().numpy()
    T_base0 = np.eye(4)
    T_base0[:3, :3] = Rotation.from_quat(qb).as_matrix()
    T_base0[:3, 3] = pb
    T_palm2base = np.linalg.inv(T_base0) @ T_palm0
    if do_metrics:
        print(f"[{slug}] T_corr(palm->base) t={np.round(T_palm2base[:3, 3] * 1000, 1)}mm")

    # root pose per env: DRO palm frame directly (verified by probe_align)
    hand_pos = torch.tensor(predict_q[:, :3], dtype=torch.float32, device=sim.device)
    hand_quat = torch.tensor([euler_xyz_to_quat_xyzw(predict_q[i, 3:6]) for i in range(n)],
                             dtype=torch.float32, device=sim.device)
    hand.write_root_pose_to_sim(torch.cat([hand_pos, hand_quat], dim=-1))
    sim.step()

    # spawn closed + settle (paper protocol)
    qclosed = torch.zeros(n, hand.num_joints, device=sim.device)
    for i in range(n):
        for j in range(16):
            qclosed[i, dof_map[j]] = float(np.clip(predict_q[i, 6 + j], -2.5, 2.5))
    hand.write_joint_state_to_sim(qclosed, torch.zeros_like(qclosed))
    for _ in range(60):
        hand.write_joint_position_to_sim(qclosed)
        sim.step()

    mets = None
    if do_metrics:
        jp_actual = hand.data.joint_pos.cpu().numpy()
        sim_closed = np.stack([jp_actual[:, dof_map[j]] for j in range(16)], axis=-1)
        err = np.abs(sim_closed - predict_q[:, 6:22]).max()
        print(f"[{slug}] closed-joint max err: {err:.4f} rad")
        body_pos = hand.data.body_pos_w.cpu().numpy()
        body_quat = hand.data.body_quat_w.cpu().numpy()
        link_pcs = np.load(LINK_PCS_NPZ)
        np_pts = sum(link_pcs[ln].shape[0] for ln in link_order)
        sim_pcs = np.zeros((n, np_pts, 3), dtype=np.float32)
        for i in range(n):
            off = 0
            for link in link_order:
                pts = link_pcs[link].astype(np.float32)
                if link == "palm_lower":
                    pts = (T_palm2base[:3, :3] @ pts.T + T_palm2base[:3, 3:4]).T.astype(np.float32)
                R = quat_to_mat_xyzw(body_quat[i, body_map[link]])
                sim_pcs[i, off:off + pts.shape[0]] = np.matmul(R, pts.T).T + body_pos[i, body_map[link]]
                off += pts.shape[0]
        align_per_env = np.zeros(n)
        for i in range(n):
            a = sim_pcs[i] - sim_pcs[i].mean(0)
            b = fk_pc[i] - fk_pc[i].mean(0)
            align_per_env[i] = np.linalg.norm(a - b, axis=-1).mean()
        align_err = float(align_per_env.mean())
        # CORA/pen from FK pcs (pure geometry, unaffected by spawn depenetration deflection)
        prox = trimesh.proximity.ProximityQuery(mesh)
        cora_frac = np.zeros(n)
        cora_mean_mm = np.zeros(n)
        pen_max_mm = np.zeros(n)
        for i in range(n):
            res = prox.on_surface(fk_pc[i])
            d = np.asarray(res[1]).ravel()
            signed = np.asarray(prox.signed_distance(fk_pc[i])).ravel()
            pen_max_mm[i] = float(max(0.0, signed.max()) * 1000)
            cora_frac[i] = (d <= 0.005).mean()
            cora_mean_mm[i] = d.mean() * 1000
        per_link_off = {}
        off = 0
        for link in link_order:
            npts = link_pcs[link].shape[0]
            seg = sim_pcs[0, off:off + npts]
            fkseg = fk_pc[0, off:off + npts]
            per_link_off[link] = float(np.linalg.norm(seg.mean(0) - fkseg.mean(0)) * 1000)
            off += npts
        print(f"[{slug}] PER-LINK centroid offset (mm), grasp 0: "
              f"{dict(sorted(per_link_off.items(), key=lambda kv: -kv[1])[:8])}")
        mets = dict(align_err_mm=align_err * 1000, cora_frac=cora_frac, cora_mean_mm=cora_mean_mm,
                    pen_max_mm=pen_max_mm, per_link_off=per_link_off)
    else:
        # still build sim pcs for nothing — skip metrics entirely
        pass

    # hold test: 1 s of gravity 0.5 (only object falls; hand links disable_gravity=True)
    p_ref = obj.data.root_pos_w[:, :3].clone()
    for _ in range(120):
        hand.write_joint_position_to_sim(qclosed)
        sim.step()
    p_end = obj.data.root_pos_w[:, :3].clone()
    hold_disp = (p_end - p_ref).norm(dim=-1).cpu().numpy()

    lift_rise = None
    if do_metrics:
        start_obj_z = obj.data.root_pos_w[:, 2].clone()
        hand_z0 = hand.data.root_pos_w[:, 2].clone()
        lift = 0.15
        for step in range(240):
            z = hand_z0 + lift * (step + 1) / 240.0
            pose = torch.cat([hand_pos, hand_quat], dim=-1).clone()
            pose[:, 2] = z
            hand.write_root_pose_to_sim(pose)
            sim.step()
        end_obj_z = obj.data.root_pos_w[:, 2].clone()
        lift_rise = (end_obj_z - start_obj_z).cpu().numpy()
    sim.clear_instance() if hasattr(sim, "clear_instance") else None
    return mets, hold_disp, lift_rise


def eval_object(slug):
    t0 = time.time()
    hold_all = []
    mets = None
    lift0 = None
    for k, d in enumerate(DIRECTIONS):
        g = tuple(0.5 * x for x in d)
        m, disp, lift = run_direction(slug, g, do_metrics=(k == 0))
        hold_all.append(disp)
        if k == 0:
            mets, lift0 = m, lift
        print(f"[{slug}] dir {d}: disp mm median={np.median(disp) * 1000:.1f} max={disp.max() * 1000:.1f}")
    hold_all = np.stack(hold_all)  # (6, n)
    success = (hold_all < 0.02).all(axis=0)
    cora_frac = mets["cora_frac"]
    res = {
        "object": slug,
        "n_grasps": hold_all.shape[1],
        "align_err_mm": mets["align_err_mm"],
        "success_rate": float(success.mean()),
        "success_n": int(success.sum()),
        "mean_cora_frac": float(cora_frac.mean()),
        "mean_cora_dist_mm": float(mets["cora_mean_mm"].mean()),
        "mean_pen_depth_mm": float(mets["pen_max_mm"].mean()),
        "max_pen_depth_mm": float(mets["pen_max_mm"].max()),
        "hold_disp_median_mm": float(np.median(hold_all) * 1000),
        "lift_rise_median_mm": float(np.median(lift0) * 1000) if lift0 is not None else None,
        "time_s": round(time.time() - t0, 1),
    }
    os.makedirs(args_cli.out_dir, exist_ok=True)
    with open(f"{args_cli.out_dir}/{slug}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "object"}, indent=2))
    return res


def main():
    if args_cli.objects:
        slugs = [s.strip() for s in args_cli.objects.split(",")]
    else:
        slugs = list(OBJECTS.keys())
    summary = []
    for slug in slugs:
        try:
            summary.append(eval_object(slug))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{slug}] FAILED: {e}")
    n = len(summary)
    if n:
        s = {
            "total_grasps": int(np.sum([r["n_grasps"] for r in summary])),
            "success_rate": float(np.mean([r["success_rate"] for r in summary])),
            "mean_penetration_mm": float(np.mean([r["mean_pen_depth_mm"] for r in summary])),
            "mean_cora_frac": float(np.mean([r["mean_cora_frac"] for r in summary])),
            "objects": summary,
        }
        with open(f"{args_cli.out_dir}/summary.json", "w") as f:
            json.dump(s, f, indent=2)
        print(json.dumps({k: v for k, v in s.items() if k != "objects"}, indent=2))
    simulation_app.close()


if __name__ == "__main__":
    main()
