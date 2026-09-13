"""Random agent on Isaac-Dexsuite-Kuka-LEAP-Reorient-v0 + DexGraspNet 2.0 ghost hands.

Same scene as kuka_leap_dexsuite_random_agent_bodex.py, but the 16 grasp ghosts
come from DexGraspNet 2.0 (dgn2_cube_gen.py -> plan_dgn2_leap_grasp_multi.py)
instead of BODex. The plan npz has the same schema (grasp_root pos+wxyz,
hand_q, hand_spheres, object_pose), so the visualization/randomization/desk
logic is identical. The random agent drives the real KUKA-LEAP robot untouched.

Usage:
  /home/andy/env_isaaclab/bin/python scripts/kuka_leap_dexsuite_random_agent_dgn2.py \
      --task Isaac-Dexsuite-Kuka-LEAP-Reorient-v0 --num_envs 1 --viz Kit \
      [--randomize_object]
"""
import argparse
import contextlib
import sys

import numpy as np
import torch


def _quat_mul_np(qa, qb):
    """Hamilton product of two wxyz quaternions (numpy arrays)."""
    wa, xa, ya, za = qa
    wb, xb, yb, zb = qb
    return np.array([
        wa * wb - xa * xb - ya * yb - za * zb,
        wa * xb + xa * wb + ya * zb - za * yb,
        wa * yb - xa * zb + ya * wb + za * xb,
        wa * zb + xa * yb - ya * xb + za * wb,
    ])

from isaaclab.app import AppLauncher

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Random agent + BODex grasp ghost visualization.")
# IMPORTANT (this script only; the plain random_agent does not need it):
# The BODex ghost is a SECOND articulation spawned AFTER the env scene was
# created. With Fabric enabled, PhysX writes ghost states to GPU fabric while
# the RTX renderer reads fabric transforms — a FabricMapper paths/attributes
# size mismatch occurs and produces CUDA_ERROR_ILLEGAL_ADDRESS + core dump in
# --viz mode (headless is unaffected). We handled it by registering the ghost
# BEFORE env.reset() (so its physx view is created in the same reset pass as
# the scene). Fabric stays ON so rendering follows physics. If you still see
# CUDA_ERROR_ILLEGAL_ADDRESS in the kit window, pass --disable_fabric as a
# fallback (visual-only; physics state won't be mirrored to the render layer).
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--plan_path", type=str, default="/home/andy/research/bodex_dexsuite/plan_dgn2_multi.npz")
parser.add_argument("--randomize_object", action="store_true", default=False,
                    help="Randomize the object pose (and transform all grasps with it) every run")
parser.add_argument("--ghost_material", type=str, default="ghost", help="ghost | opaque")
parser.add_argument(
    "--ghost_hand", type=str, default="leap", choices=["leap", "allegro"],
    help="Which hand model the BODex grasp poses are shown on (leap=LEAP_HAND_CFG,"
    " allegro=BODex-derived Allegro USD).",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main():
    """Random actions agent with BODex ghost robot in the scene."""
    torch.manual_seed(42)

    visualizer = list(args_cli.visualizer or [])
    if args_cli.headless:
        visualizer = []
    args_cli.visualizer = visualizer

    # Launch Kit BEFORE importing the task package (pxr import safety).
    app_launcher = AppLauncher(args_cli)
    sim_app = app_launcher.app

    import LEAP_Isaaclab.tasks  # noqa: F401  <-- registers Isaac-Dexsuite-Kuka-LEAP-*

    import gymnasium as gym
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    # parse configuration via Hydra
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False

    # --------------------------------------------------- BODex ghost cfg ----------
    # 16 FLOATING LEAP HANDS (no arm), one per diverse BODex grasp, all fixed in
    # world space around the object. Registered as scene entities so PhysX
    # creates their articulation views in the SAME PHYSICS_READY pass as the
    # environment (keeps fabric consistent; prevents CUDA_ERROR_ILLEGAL_ADDRESS).
    plan = np.load(args_cli.plan_path)
    # multi plan: grasp_root [N,7], hand_q [N,16]
    grasp_roots = np.atleast_2d(plan["grasp_root"])   # handle both N and 1
    hand_qs = np.atleast_2d(plan["hand_q"])
    object_pose = plan["object_pose"].copy()          # [7] pos + wxyz quat (curobo)
    n_ghosts = grasp_roots.shape[0]

    # --- optional object randomization ----------------------------------------
    # The BODex grasps were solved relative to the fixed object pose. If the user
    # wants the object to appear at a random pose every run, we must transform
    # BOTH the object AND all grasp_root poses by the same relative delta, so the
    # hand-object relationship stays exactly as the solver produced it.
    # Shared desk geometry + helpers (used by both randomization and sanity check):
    TABLE_C = np.array([-0.55, 0.0, 0.235])          # from BODex TABLE dict
    TABLE_HALF = np.array([0.4, 0.75, 0.02])
    TABLE_TOP = TABLE_C[2] + TABLE_HALF[2]           # 0.255 (desk top face)
    OBJ_HALF_AXES = np.array([0.025, 0.05, 0.05])    # cube half-extents (0.05x0.10x0.10)

    def _sphere_box_pen(center, r, box_c, box_half):
        d = np.abs(center - box_c) - box_half
        if (d < 0).all():
            return r + d.max()
        return r - np.sqrt((np.maximum(d, 0.0) ** 2).sum())

    if args_cli.randomize_object:
        # random pose: position within a cube around the original, any rotation
        # (uniform quaternion via the sphere method; sampled with numpy RNG)
        rng = np.random.default_rng(int(np.random.randint(0, 2**31)))  # new seed per run
        from scipy.spatial.transform import Rotation as _Rst

        # hand collision spheres from the plan (world frame for the ORIGINAL
        # object pose, [N, n_spheres, 4]); transformed with the object delta.
        hand_spheres = plan["hand_spheres"] if "hand_spheres" in plan.files else None
        if hand_spheres is not None:
            hand_spheres = np.asarray(hand_spheres)

        best = None
        for _attempt in range(400):
            dx, dy, dz = rng.uniform(-0.35, 0.35, size=3)
            new_obj_pos = np.array([object_pose[0] + dx, object_pose[1] + dy,
                                    np.clip(object_pose[2] + dz * 0.4, 0.30, 1.0)])
            q = rng.normal(size=4)
            q = q / np.linalg.norm(q)                  # wxyz
            R_old = _Rst.from_quat(np.roll(object_pose[3:7], -1)).as_matrix()
            R_new = _Rst.from_quat(np.roll(q, -1)).as_matrix()
            R_delta = R_new @ R_old.T
            t_old = object_pose[:3]

            # 1) object itself: exact rotated OBB extent in z must clear the desk
            dz_obj = np.abs(R_new[2]) @ OBJ_HALF_AXES
            obj_low = new_obj_pos[2] - dz_obj
            if obj_low < TABLE_TOP + 0.01:             # contact/tangent rejected
                continue

            # 2) every grasp: transform its collision spheres, require all clear
            if hand_spheres is not None:
                safe = True
                for i in range(n_ghosts):
                    sp = hand_spheres[i]
                    p_new = (R_delta @ (sp[:, :3] - t_old).T).T + new_obj_pos
                    pen = max((_sphere_box_pen(c, r, TABLE_C, TABLE_HALF)
                               for c, r in zip(p_new, sp[:, 3])), default=-1e9)
                    if pen > 0:
                        safe = False
                        break
            else:
                # fallback: palm-center only (plan without hand_spheres)
                safe = True
                for i in range(n_ghosts):
                    p_new = R_delta @ (grasp_roots[i][:3] - t_old) + new_obj_pos
                    if p_new[2] < TABLE_TOP + 0.05:
                        safe = False
                        break
            if safe:
                best = (new_obj_pos, q, R_delta, R_old, t_old)
                break
        if best is None:
            raise RuntimeError("Could not find a desk-safe random object pose in 400 tries")
        new_obj_pos, q, R_delta, R_old, t_old = best
        # apply delta to every grasp root (rotate about object center, then translate)
        for i in range(n_ghosts):
            p_old = grasp_roots[i][:3]
            p_new = R_delta @ (p_old - t_old) + new_obj_pos
            q_old = grasp_roots[i][3:7]
            q_delta = _Rst.from_matrix(R_delta).as_quat()   # xyzw
            q_delta_w = np.roll(q_delta, 1)                 # -> wxyz
            q_new = _quat_mul_np(q_delta_w, q_old)          # wxyz Hamilton product
            grasp_roots[i] = np.concatenate([p_new, q_new])
        # object itself now at the new pose (used for the pinned obj_bodex_root)
        object_pose[:3] = new_obj_pos
        object_pose[3:7] = q
        print(f"[BODex] object randomized (desk-safe): pos {np.round(new_obj_pos, 3)} "
              f"quat(wxyz) {np.round(q, 3)}  -- grasps transformed with the object")
    elif "hand_spheres" in plan.files:
        # not randomizing: sanity-report that the stored grasps are desk-safe
        worst = -1e9
        plane = plan["hand_spheres"]
        for i in range(n_ghosts):
            for c_r in plane[i]:
                worst = max(worst, _sphere_box_pen(c_r[:3], c_r[3], TABLE_C, TABLE_HALF))
        print(f"[BODex] desk check: worst sphere penetration = {worst:.4f} m "
              f"({'OK: no desk intersection' if worst <= 0 else 'WARNING: intersects desk!'})")

    print(f"[BODex] plan: {n_ghosts} grasps")
    for i in range(n_ghosts):
        print(f"[BODex] grasp {i}: root {np.round(grasp_roots[i][:3], 3)}  "
              f"q {np.round(hand_qs[i], 3)}")

    import isaaclab.sim as sim_utils
    from isaaclab.assets.articulation import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg as _ImpAct
    from LEAP_Isaaclab.assets import LEAP_HAND_CFG

    # cuRobo convention is wxyz; Isaac Lab 3.0 uses xyzw
    from isaaclab.utils.math import convert_quat, quat_mul

    # Which hand to display: LEAP (default) or BODex-derived Allegro USD.
    if args_cli.ghost_hand == "allegro":
        # Clean Allegro hand USD converted from DGN2's Isaac-Gym URDF + meshes
        # (BODex pack's OBJ meshes fail the usdex topology check; DGN2's are OK).
        _HAND_CFG = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/BODex_phantom",
            spawn=sim_utils.UsdFileCfg(
                usd_path=(
                    "/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab/source/"
                    "LEAP_Isaaclab/LEAP_Isaaclab/assets/allegro_hand_v2/kuka_allegro.usda"
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=True,
                    retain_accelerations=False,
                    enable_gyroscopic_forces=False,
                    angular_damping=0.01,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=64 / np.pi * 180.0,
                    max_depenetration_velocity=1000.0,
                    max_contact_impulse=1e32,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0005,
                    fix_root_link=False,
                ),
            ),
            actuators={
                "fingers": _ImpAct(
                    joint_names_expr=[".*"],
                    effort_limit=0.5,
                    velocity_limit=100.0,
                    stiffness=0.0,
                    damping=1.0,
                    armature=1e-3,
                    friction=0.01,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )
        # BODex joint_0.0..15.0 -> DGN2 USD names (index/middle/ring/thumb x4)
        _JOINT_ORDER = [
            "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
            "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
            "ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3",
            "thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
        ]
    else:
        # LEAP ghost: use the USD converted FROM THE SAME URDF the BODex solver
        # used (leap_own_description/leap_hand_right.urdf). Joint names appear
        # as tn__0_..tn__15_ in USD traversal order, which matches the curobo
        # hand_q order exactly (0..15) -> NO reordering needed. Geometry is
        # identical, so the displayed hands are exactly what BODex solved
        # (the old leap_hand_v1 USD was a different model version and produced
        # incorrect finger poses).
        _HAND_CFG = LEAP_HAND_CFG.replace(
            spawn=LEAP_HAND_CFG.spawn.replace(
                usd_path=(
                    "/home/andy/Documents/projects/LEAP_Hand_Isaac_Lab/source/"
                    "LEAP_Isaaclab/LEAP_Isaaclab/assets/leap_bodex_usd2/leap_right.usda"
                ),
            ),
        )
        # articulation_props/collision etc. are handled below per-ghost
        _JOINT_ORDER = None        # vector order == hand_q order
        _JOINT_QIDX = None
        _HAND_CFG.soft_joint_pos_limit_factor = 1.3

    for i in range(n_ghosts):
        rot_xyzw = convert_quat(torch.tensor(grasp_roots[i][3:7], dtype=torch.float32), to="xyzw").numpy()
        jp = {}
        # NOTE: init_state.joint_pos must satisfy HARD PhysX limits. Different
        # joints have different limits (dip [-0.366,2.042], mcp [-0.314,2.23],
        # a_8 [-1.047,1.047]) and BODex-solved values may exceed them. 0 always
        # lies inside every limit, so we place 0.0 here only to pass
        # _validate_cfg; the true BODex qs are written right after env.reset()
        # via write_joint_state_to_sim (no validation there).
        if _JOINT_ORDER is not None:
            for k, jn in enumerate(_JOINT_ORDER):
                jp[jn] = 0.0
        else:
            # LEAP bodex USD: zero every tn__* joint (regex matches all; 0 is
            # always inside every limit, so this passes _validate_cfg). The
            # bodex-converted USD is the only LEAP case with no _JOINT_ORDER.
            jp["tn__.*"] = 0.0
        ghost_cfg = _HAND_CFG.replace(
            prim_path=f"{{ENV_REGEX_NS}}/BODex_phantom_{i}",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(grasp_roots[i][:3].tolist()),
                rot=tuple(rot_xyzw.tolist()),
                joint_pos=jp,
            ),
        )
        # Floating hand registered with the env scene; gravity disabled and
        # collision DISABLED: never interacts with the robot/object, so its
        # pose is exactly the BODex grasp pose.
        ghost_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,
            contact_offset=0.001,
            rest_offset=0.0,
        )
        ghost_cfg.spawn.articulation_props.fix_root_link = False
        setattr(env_cfg.scene, f"bodex_phantom_{i}", ghost_cfg)

    # Pin the object at the BODex grasp pose too: the BODex grasps were solved
    # for the cube (0.05x0.1x0.1) at OBJECT_POSE; spawn exactly that cube and
    # hold it there so the floating hands visually align with the object.
    from isaaclab_tasks.manager_based.manipulation.dexsuite.dexsuite_env_cfg import ObjectCfg
    env_cfg.scene.object.spawn = ObjectCfg().cube
    # object_pose is the (possibly randomized) pose computed in the plan block above
    obj_rot_xyzw = convert_quat(torch.tensor(object_pose[3:7], dtype=torch.float32), to="xyzw").numpy()
    obj_bodex_root = torch.tensor(
        [object_pose[0], object_pose[1], object_pose[2],
         obj_rot_xyzw[0], obj_rot_xyzw[1], obj_rot_xyzw[2], obj_rot_xyzw[3]],
        dtype=torch.float32, device=env_cfg.sim.device,
    ).unsqueeze(0)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = env.unwrapped
    obj = env.scene["object"]

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    ghosts = [env.scene[f"bodex_phantom_{i}"] for i in range(n_ghosts)]
    ghost_qs = []
    ghost_roots = []
    # Quaternion handling -- VERIFIED by direct fingertip comparison probe
    # (scripts/probe_ghost_tips.py): curobo's grasp_root is ALREADY the URDF
    # base-frame pose; only reorder wxyz -> Isaac Lab 3.0 xyzw. The
    # hand_pose_transfer rotation must NOT be applied (it made the tips drift
    # 0.19 m away; without it they match curobo FK within 0.075 m).
    for i in range(n_ghosts):
        rot_xyzw = convert_quat(torch.tensor(grasp_roots[i][3:7], dtype=torch.float32), to="xyzw")
        if _JOINT_ORDER is not None:
            # allegro/leap: build vector in the Articulation's resolved joint order
            names = ghosts[i].data.joint_names
            vals = []
            for jn in names:
                if args_cli.ghost_hand == "leap":
                    # LEAP USD traversal (a_1,a_0,a_2,...) -> BODex index via
                    # chain mapping (_JOINT_QIDX is aligned to _JOINT_ORDER)
                    if jn in _JOINT_ORDER:
                        vals.append(float(hand_qs[i][_JOINT_QIDX[_JOINT_ORDER.index(jn)]]))
                    else:
                        vals.append(0.0)
                else:
                    # allegro: match mangled USD name to BODex index (00->0, 10->1)
                    digits = jn.split("joint_")[-1].split("_")[0] if "joint_" in jn else ""
                    digits = digits.lstrip("0") or "0"
                    idx = int(digits)
                    vals.append(float(hand_qs[i][idx]) if idx < len(hand_qs[i]) else 0.0)
            tgt = torch.tensor(vals, dtype=torch.float32, device=env.device).unsqueeze(0)
        else:
            tgt = torch.tensor(hand_qs[i], dtype=torch.float32, device=env.device).unsqueeze(0)
        ghost_qs.append(tgt)
        ghost_roots.append(torch.tensor(
            [grasp_roots[i][0], grasp_roots[i][1], grasp_roots[i][2],
             rot_xyzw[0], rot_xyzw[1], rot_xyzw[2], rot_xyzw[3]],
            dtype=torch.float32, device=env.device,
        ).unsqueeze(0))

    # ------------------------------------------------------------ env reset ----
    env.reset()
    # The ghosts (scene entities) are initialized with the env. Pin each root
    # at its BODex grasp pose and hold the finger joints.
    for i in range(n_ghosts):
        ghosts[i].write_root_pose_to_sim(ghost_roots[i])
        ghosts[i].write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((ghosts[i].num_instances, 6), device=env.device))
        ghosts[i].write_joint_state_to_sim(ghost_qs[i], torch.zeros_like(ghost_qs[i]))
        ghosts[i].write_data_to_sim()
    print(f"[BODex] {n_ghosts} ghost {args_cli.ghost_hand} hands initialized (registered + callback fired)")

    # ---- quaternion-convention regression check (Isaac Lab 3.0 = xyzw) ----
    # The expected order is [x, y, z, w] (scalar last). If this check fails,
    # the convention changed again or the conversion above is wrong.
    with torch.inference_mode():
        n_ok = 0
        for i in range(n_ghosts):
            q_expected = ghost_roots[i][0, 3:7]  # xyzw we wrote
            q_readback = ghosts[i].data.root_quat_w[0]  # what sim reports
            ok = bool(torch.allclose(q_expected, q_readback, atol=1e-3))
            if ok:
                n_ok += 1
            else:
                print(
                    f"[QUAT-CHECK] ghost {i} MISMATCH! wrote(xyzw)={q_expected.tolist()} "
                    f"readback={q_readback.tolist()}"
                )
        print(f"[QUAT-CHECK] {n_ok}/{n_ghosts} ghosts: root_quat_w matches written xyzw " 
              f"(atol=1e-3); Isaac Lab 3.0 convention = [x, y, z, w]")

    # make the ghosts translucent (blue) for visual separation
    if args_cli.ghost_material == "ghost":
        from pxr import Usd, UsdShade, Sdf
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
        from isaaclab.sim.spawners.materials import spawn_preview_surface
        stage = sim_utils.SimulationContext.instance().stage
        for i in range(n_ghosts):
            ghost_prim = stage.GetPrimAtPath(f"/World/envs/env_0/BODex_phantom_{i}")
            mat_path = f"/World/envs/env_0/BODex_phantom_{i}/materials/GhostMat"
            mat = spawn_preview_surface(
                mat_path,
                PreviewSurfaceCfg(diffuse_color=(0.35, 0.55, 1.0), metallic=0.1, roughness=0.35, opacity=0.4),
            )
            n_bound = 0
            stack = [ghost_prim]
            while stack:
                p = stack.pop()
                if p and p.IsValid() and p.GetTypeName() == "Mesh":
                    rel = p.CreateRelationship(UsdShade.Tokens.materialBinding)
                    rel.AddTarget(Sdf.Path(mat_path))
                    n_bound += 1
                if p and p.IsValid():
                    stack.extend(p.GetFilteredChildren(Usd.PrimDefaultPredicate))
            print(f"[BODex] ghost {i} translucent ({n_bound} meshes bound)")

    # Silence the goal_pose command marker (the red/cyan cluster) so it is not
    # mistaken for the object. It is only a visualization gimmick that stays at
    # the goal pose while the object itself is the real cube mesh.
    try:
        from isaaclab_tasks.manager_based.manipulation.dexsuite import mdp as dexsuite_mdp2
        import isaaclab.managers as mgrs
        # CommandTermCfg.debug_vis toggles the marker; default True.
        # Instead of changing config after env already built, hide the prims directly.
        stage = sim_utils.SimulationContext.instance().stage
        for vis_path in ["/Visuals/Command/goal_pose", "/Visuals/Command/body_pose", "/Visuals/Command"]:
            prim = stage.GetPrimAtPath(vis_path)
            if prim:
                from isaaclab.sim.utils.prims import set_prim_visibility
                set_prim_visibility(prim, False)
                print(f"[VIS] hidden {vis_path}")
    except Exception as e:
        print(f"[VIS] could not hide goal_pose marker: {e}")

    # ------------------------------------------------------------ random loop ----
    sim = env.unwrapped.sim
    num_steps = 0
    max_steps = args_cli.num_envs * 200 if args_cli.num_envs else 200
    while True:
        if sim.visualizers:
            if not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                break
        elif num_steps >= max_steps:
            print(f"[INFO]: headless mode: reached {num_steps} steps, exiting.")
            break
        # Pin the ghosts BEFORE env.step(): physx integrates after this write, so
        # the roots never accumulate velocity/drift and the render layer (which
        # consumes physx state each step) sees exactly the BODex poses.
        for i in range(n_ghosts):
            ghosts[i].write_root_pose_to_sim(ghost_roots[i])
            ghosts[i].write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((ghosts[i].num_instances, 6), device=env.device))
            # write joints as a *state*, not a torque target: set_joint_position_target
            # applies actuator stiffness and the reaction torques rotate the free
            # root link a few degrees per step (the root_quat drift seen in-chat).
            ghosts[i].write_joint_state_to_sim(ghost_qs[i], torch.zeros_like(ghost_qs[i]))
            ghosts[i].write_data_to_sim()
        if num_steps % 50 == 0:
            with torch.inference_mode():
                # Immediately after the write, data.root_quat_w mirrors the pose we
                # just set (the physx buffer was invalidated and re-read via
                # write_data_to_sim). We intentionally do NOT measure after
                # env.step(): the physics integrator can move a free-floating root.
                for i in range(n_ghosts):
                    ok = bool(torch.allclose(
                        ghost_roots[i][0, 3:7], ghosts[i].data.root_quat_w[0], atol=1e-3))
                    if not ok:
                        print(
                            f"[QUAT-CHECK] step {num_steps} ghost {i} DRIFTED: "
                            f"expected={ghost_roots[i][0, 3:7].tolist()} "
                            f"got={ghosts[i].data.root_quat_w[0].tolist()}"
                        )
        # hold the object at the BODex grasp pose too (visual alignment)
        obj.write_root_pose_to_sim(obj_bodex_root)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((obj.num_instances, 6), device=env.device))
        obj.write_data_to_sim()
        with torch.inference_mode():
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            env.step(actions)
        if num_steps % 10 == 0:
            gpos = ghosts[0].data.root_pos_w.cpu().numpy().round(4)
            print(
                f"[STEP] step {num_steps:3d} | obj pos: {obj.data.root_pos_w.cpu().numpy().round(4)} | "
                f"ghost0 root: {gpos} | ghosts: {n_ghosts}"
            )
        num_steps += 1

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
