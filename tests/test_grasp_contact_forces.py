"""CPU-only tests at the grasp task's external contact-sensor boundary."""

import importlib.util
import sys
import torch
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE = Path(__file__).resolve().parents[1] / (
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_grasp/contact_forces.py"
)
spec = importlib.util.spec_from_file_location("grasp_contact_forces", MODULE)
contacts = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = contacts
spec.loader.exec_module(contacts)


def sensor_fixture():
    """Stand-in for the public Isaac Lab ProxyArray/data interface, not simulator internals."""
    normal = torch.tensor([[[[0.0, 0.0, 2.0]]], [[[0.0, 0.0, 0.0]]]])
    friction = torch.tensor([[[[0.5, 0.0, 0.0]]], [[[0.0, 0.0, 0.0]]]])

    def proxy(value):
        return SimpleNamespace(torch=value)

    return SimpleNamespace(
        data=SimpleNamespace(
            normal_force_matrix_w=proxy(normal),
            friction_force_matrix_w=proxy(friction),
            normal_force_matrix_w_history=proxy(torch.stack((normal, normal * 0.5), dim=1)),
            friction_force_matrix_w_history=proxy(torch.stack((friction, friction * 0.5), dim=1)),
            contact_pos_w=proxy(torch.tensor([[[[1.0, 2.0, 3.0]]], [[[9.0, 9.0, 9.0]]]])),
        )
    )


class ContactSampleTests(unittest.TestCase):
    def test_returns_full_force_on_sensing_body_with_history_and_position_validity(self):
        sensor = sensor_fixture()
        sample = contacts.read_contact_sample(sensor, 2, 1 / 120)
        torch.testing.assert_close(sample.normal, torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]))
        torch.testing.assert_close(sample.friction, torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        torch.testing.assert_close(sample.total, torch.tensor([[0.5, 0.0, 2.0], [0.0, 0.0, 0.0]]))
        torch.testing.assert_close(sample.history_normal[0], torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]]))
        torch.testing.assert_close(sample.history_friction[0], torch.tensor([[0.5, 0.0, 0.0], [0.25, 0.0, 0.0]]))
        self.assertEqual(sample.position_valid.tolist(), [True, False])
        torch.testing.assert_close(sample.position[0], torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.isnan(sample.position[1]).all())
        sensor.data.normal_force_matrix_w.torch.zero_()
        sensor.data.normal_force_matrix_w_history.torch.zero_()
        self.assertEqual(sample.normal[0, 2].item(), 2.0)
        self.assertEqual(sample.history_normal[0, 0, 2].item(), 2.0)

    def test_missing_split_or_history_fails_instead_of_using_legacy_aliases(self):
        fields = (
            "normal_force_matrix_w",
            "friction_force_matrix_w",
            "normal_force_matrix_w_history",
            "friction_force_matrix_w_history",
            "contact_pos_w",
        )
        for field in fields:
            for missing_kind in ("none", "absent", "unsupported"):
                with self.subTest(field=field, missing_kind=missing_kind):
                    sensor = sensor_fixture()
                    sensor.data.force_matrix_w = torch.ones(2, 1, 1, 3)
                    if missing_kind == "none":
                        setattr(sensor.data, field, None)
                    elif missing_kind == "absent":
                        delattr(sensor.data, field)
                    else:

                        def unavailable(_):
                            raise NotImplementedError("backend does not implement this")

                        data_type = type("UnsupportedData", (), {field: property(unavailable)})
                        data = data_type()
                        data.__dict__.update(vars(sensor.data))
                        sensor.data = data
                    with self.assertRaisesRegex(RuntimeError, field):
                        contacts.read_contact_sample(sensor, 2, 1 / 120)

    def test_support_history_preserves_each_tick_and_does_not_cancel_opposing_partners(self):
        sensor = sensor_fixture()
        normal = torch.tensor([[[[[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]])
        friction = torch.tensor([[[[[0.0, 4.0, 0.0], [0.0, -4.0, 0.0]]], [[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]]]])
        sensor.data.normal_force_matrix_w_history = normal
        sensor.data.friction_force_matrix_w_history = friction
        torch.testing.assert_close(contacts.read_support_load(sensor, 1), torch.tensor([[10.0, 2.0]]))
        for field in ("normal_force_matrix_w_history", "friction_force_matrix_w_history"):
            sensor = sensor_fixture()
            setattr(sensor.data, field, None)
            with self.assertRaisesRegex(RuntimeError, field):
                contacts.read_support_load(sensor, 2)

    def test_rejects_ambiguous_shapes_bad_history_and_nonfinite_forces(self):
        for case in ("env_count", "tip_partners", "support_bodies", "history_length", "empty_partners", "nan"):
            with self.subTest(case=case):
                sensor = sensor_fixture()
                if case == "env_count":
                    with self.assertRaisesRegex(RuntimeError, "shape"):
                        contacts.read_contact_sample(sensor, 1, 1 / 120)
                    continue
                if case == "tip_partners":
                    sensor.data.normal_force_matrix_w = torch.zeros(2, 1, 2, 3)
                elif case == "support_bodies":
                    sensor.data.normal_force_matrix_w_history = torch.zeros(2, 2, 2, 1, 3)
                elif case == "history_length":
                    sensor.data.friction_force_matrix_w_history = torch.zeros(2, 1, 1, 1, 3)
                elif case == "empty_partners":
                    sensor.data.normal_force_matrix_w_history = torch.zeros(2, 2, 1, 0, 3)
                else:
                    sensor.data.normal_force_matrix_w.torch[0, 0, 0, 0] = float("nan")
                    sensor.data.normal_force_matrix_w_history.torch[0, 0, 0, 0, 0] = float("nan")
                with self.assertRaises(RuntimeError):
                    contacts.read_contact_sample(sensor, 2, 1 / 120)
                if case != "tip_partners":
                    with self.assertRaises(RuntimeError):
                        contacts.read_support_load(sensor, 2)
        for dt in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                contacts.read_contact_sample(sensor_fixture(), 2, dt)

    def test_raw_position_ignores_zero_force_speculation_and_weights_only_loaded_points(self):
        sensor = sensor_fixture()
        sensor.contact_view = SimpleNamespace(
            get_contact_data=lambda dt: (
                torch.tensor([[1.0], [3.0], [0.0], [0.0]]),
                torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [100.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]),
                torch.zeros(4, 3),
                torch.zeros(4, 1),
                torch.tensor([[3], [1]]),
                torch.tensor([[0], [3]]),
            )
        )
        sample = contacts.read_contact_sample(sensor, 2, 1 / 120)
        torch.testing.assert_close(sample.position[0], torch.tensor([3.0, 0.0, 0.0]))
        self.assertEqual(sample.position_valid.tolist(), [True, False])
        self.assertTrue(torch.isnan(sample.position[1]).all())
        sensor.data.normal_force_matrix_w.torch[0, 0, 0].zero_()
        sample = contacts.read_contact_sample(sensor, 2, 1 / 120)
        self.assertFalse(sample.position_valid.any())

    def test_raw_contact_errors_and_truncated_buffers_do_not_fall_back_to_average(self):
        sensor = sensor_fixture()

        def broken(dt):
            raise RuntimeError("raw contact read failed")

        sensor.contact_view = SimpleNamespace(get_contact_data=broken)
        with self.assertRaisesRegex(RuntimeError, "raw contact"):
            contacts.read_contact_sample(sensor, 2, 1 / 120)
        sensor.contact_view.get_contact_data = lambda dt: (
            torch.ones(2, 1),
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            torch.zeros(2, 1),
            torch.tensor([[3], [0]]),
            torch.tensor([[0], [0]]),
        )
        with self.assertRaisesRegex(RuntimeError, "buffer"):
            contacts.read_contact_sample(sensor, 2, 1 / 120)


if __name__ == "__main__":
    unittest.main()
