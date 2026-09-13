"""Debug: spawn only the KUKA_LEAP articulation via sim prim spawner, print body schema."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(parser.parse_args(), headless=True)
sim_app = app_launcher.app

import isaaclab.sim as sim_utils
from LEAP_Isaaclab.assets import KUKA_LEAP_CFG

sim = sim_utils.SimulationContext()

# spawn robot prim at a concrete path
cfg = KUKA_LEAP_CFG.spawn
prim = sim_utils.spawn_from_urdf("/World/Robot", cfg)
print("spawned:", prim.IsValid())

stage = sim.stage
robot_prim = stage.GetPrimAtPath("/World/Robot")
print("Robot prim valid:", robot_prim.IsValid())

def walk(p, depth=0, maxd=30):
    for c in p.GetChildren():
        schemas = c.GetAppliedSchemas()
        flags = []
        if "PhysicsRigidBodyAPI" in schemas:
            flags.append("RB")
        if "PhysxContactReportAPI" in schemas:
            flags.append("CR")
        if flags:
            print("  " * depth + c.GetName() + " [" + ",".join(flags) + "]")
        if depth < maxd:
            walk(c, depth + 1, maxd)

walk(robot_prim)
print("== done")
sim_app.close()
