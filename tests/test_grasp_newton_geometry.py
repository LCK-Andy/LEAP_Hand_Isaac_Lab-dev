"""CPU USD tests for Stage A's local Newton collision schema adaptation."""

import importlib.util
import numpy as np
import unittest
from pathlib import Path
from types import SimpleNamespace

from pxr import Usd, UsdGeom, UsdPhysics

MODULE = Path(__file__).resolve().parents[1] / (
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_grasp/newton_geometry.py"
)
spec = importlib.util.spec_from_file_location("grasp_newton_geometry", MODULE)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


class Array:
    def __init__(self, data):
        self.data = np.asarray(data)

    def numpy(self):
        return self.data


class NewtonGeometryTests(unittest.TestCase):
    def make_stage(self):
        source = Usd.Stage.CreateInMemory()
        group = UsdGeom.Xform.Define(source, "/Collisions").GetPrim()
        source.SetDefaultPrim(group)
        cube = UsdGeom.Cube.Define(source, "/Collisions/box")
        cube.CreateSizeAttr(0.03)
        cube.AddTranslateOp().Set((0.01, 0.02, 0.03))
        mesh = UsdGeom.Mesh.Define(source, "/Collisions/pulp")
        mesh.CreatePointsAttr([(0, 0, 0), (0.01, 0, 0), (0, 0.01, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        UsdPhysics.CollisionAPI.Apply(group)
        UsdPhysics.MeshCollisionAPI.Apply(group).CreateApproximationAttr("convexDecomposition")
        stage = Usd.Stage.CreateInMemory()
        root = "/World/envs/env_0/Robot"
        body = UsdGeom.Xform.Define(stage, root + "/fingertip")
        body.AddTranslateOp().Set((1.0, 2.0, 3.0))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        instance = stage.DefinePrim(root + "/fingertip/collisions")
        instance.GetReferences().AddReference(source.GetRootLayer().identifier)
        instance.SetInstanceable(True)
        return source, stage, str(instance.GetPath())

    def test_deinstances_and_relocates_schema_without_changing_geometry_or_source(self):
        source, stage, path = self.make_stage()
        source_before = source.GetRootLayer().ExportToString()
        authored_before = stage.GetRootLayer().ExportToString()
        cube = UsdGeom.Cube(stage.GetPrimAtPath(path + "/box"))
        world_before = cube.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        points_before = UsdGeom.Mesh(stage.GetPrimAtPath(path + "/pulp")).GetPointsAttr().Get()
        self.assertTrue(cube.GetPrim().IsInstanceProxy())
        self.assertFalse(cube.GetPrim().HasAPI(UsdPhysics.CollisionAPI))
        result = geometry.prepare_newton_hand_collisions(stage)
        self.assertEqual(
            result,
            {
                "groups": 1,
                "colliders": 2,
                "mesh_approximations": {"convexDecomposition": 1},
                "deinstanced": 1,
            },
        )
        self.assertFalse(stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI))
        self.assertFalse(stage.GetPrimAtPath(path).HasAPI(UsdPhysics.MeshCollisionAPI))
        for name in ("box", "pulp"):
            prim = stage.GetPrimAtPath(path + "/" + name)
            self.assertFalse(prim.IsInstanceProxy())
            self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
        cube = UsdGeom.Cube(stage.GetPrimAtPath(path + "/box"))
        self.assertEqual(cube.GetSizeAttr().Get(), 0.03)
        self.assertEqual(cube.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), world_before)
        pulp = stage.GetPrimAtPath(path + "/pulp")
        self.assertEqual(UsdGeom.Mesh(pulp).GetPointsAttr().Get(), points_before)
        self.assertEqual(UsdPhysics.MeshCollisionAPI(pulp).GetApproximationAttr().Get(), "convexDecomposition")
        self.assertEqual(source.GetRootLayer().ExportToString(), source_before)
        self.assertEqual(stage.GetRootLayer().ExportToString(), authored_before)
        self.assertFalse(stage.GetSessionLayer().empty)

    def test_empty_collision_groups_fail_instead_of_importing_collisionless_hand(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/fingertip/collisions")
        with self.assertRaisesRegex(RuntimeError, "no geometry"):
            geometry.prepare_newton_hand_collisions(stage)

    def test_body_filter_matches_without_colliders_are_rejected(self):
        model = SimpleNamespace(body_label=["hand", "object"], shape_body=Array([1]), shape_flags=Array([2]))
        sensor = SimpleNamespace(
            contact_view=SimpleNamespace(
                sensing_type="body",
                counterpart_type="body",
                sensing_indices=[0],
                counterpart_indices=[[1]],
            )
        )
        with self.assertRaisesRegex(RuntimeError, "hand"):
            geometry.validate_newton_collision_coverage(model, [sensor])
        model.shape_body = Array([0, 1])
        model.shape_flags = Array([0, 2])
        with self.assertRaisesRegex(RuntimeError, "hand"):
            geometry.validate_newton_collision_coverage(model, [sensor])
        model.shape_flags = Array([2, 2])
        self.assertEqual(
            geometry.validate_newton_collision_coverage(model, [sensor]),
            {
                "required_bodies": 2,
                "colliding_shapes": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
