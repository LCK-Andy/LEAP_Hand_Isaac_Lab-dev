"""Probe: disable fingertip sensors, build env, then test contact-view patterns."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(parser.parse_args(), headless=True)
sim_app = app_launcher.app

import isaaclab.sim as sim
import isaaclab_tasks  # noqa: F401
import LEAP_Isaaclab.tasks  # noqa: F401

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import gymnasium as gym

gym_id = "Isaac-Dexsuite-Kuka-LEAP-Reorient-v0"
env_cfg = parse_env_cfg(gym_id, device="cuda:0", num_envs=2)
# disable contact sensors so env builds (we probe views manually)
for sensor_name in ["fingertip_object_s", "fingertip_2_object_s", "fingertip_3_object_s", "thumb_fingertip_object_s"]:
    setattr(env_cfg.scene, sensor_name, None)
# disable the contact-force observation term too
env_cfg.observations.proprio.contact = None
env = gym.make(gym_id, cfg=env_cfg)

from omni.physics.tensors import create_simulation_view

sview = create_simulation_view("torch")

DP = "/World/envs/env_*/Robot/Geometry/iiwa7_base_link/iiwa7_link_0/iiwa7_link_1/iiwa7_link_2/iiwa7_link_3/iiwa7_link_4/iiwa7_link_5/iiwa7_link_6/iiwa7_link_7/base/palm_lower/mcp_joint/pip/dip/fingertip"

variants = [
    ("A leaf-only str no filter", "/World/envs/env_*/Robot/(fingertip)", []),
    ("B leaf-only str + filter", "/World/envs/env_*/Robot/(fingertip)", ["/World/envs/env_*/Object_*"]),
    ("C ** leaf no filter", "/World/envs/env_*/Robot/**/(fingertip)", []),
    ("D ** leaf + filter", "/World/envs/env_*/Robot/**/(fingertip)", ["/World/envs/env_*/Object_*"]),
    ("E ** before leaf no filter", "/World/envs/env_*/Robot/**/fingertip", []),
    ("F deep path no filter", DP, []),
    ("G deep path + filter", DP, ["/World/envs/env_*/Object_*"]),
    ("H deep path + mesh suffix", DP + "/fingertip/fingertip", []),
]

variants += [
    ("I list-sensor + nested-filters", ["/World/envs/env_*/Robot/**/(fingertip)"], [["/World/envs/env_*/Object"]]),
    ("J list-sensor + nested-filters-per-env", ["/World/envs/env_*/Robot/**/(fingertip)"], [["/World/envs/env_0/Object", "/World/envs/env_1/Object"]]),
    ("K list-sensor + generated-per-env", ["/World/envs/env_*/Robot/**/(fingertip)"], [[f"/World/envs/env_{i}/Object" for i in range(2)]]),
    ("M exact-path + generated-per-env", [DP], [[f"/World/envs/env_{i}/Object" for i in range(2)]]),
]

for label, sensor_pat, filters in variants:
    try:
        v = sview.create_rigid_body_view(sensor_pat)
        b_count = str(v.count)
    except Exception as e:  # noqa: BLE001
        b_count = f"FAIL {str(e)[:40]}"
    try:
        v = sview.create_rigid_contact_view(sensor_pat, filter_patterns=filters, max_contact_data_count=16)
        c_ok = f"OK sensor_count={v.sensor_count} filter_count={v.filter_count}"
    except Exception as e:  # noqa: BLE001
        c_ok = f"FAIL {str(e)[:70]}"
    print(f":: {label}\n   body={b_count}  contact={c_ok}")
print(":: done")
sim_app.close()
