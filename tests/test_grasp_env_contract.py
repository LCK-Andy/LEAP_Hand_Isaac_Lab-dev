"""CPU regression at the real grasp environment's sensor/observation boundary.

Only simulator assets/sensors and identity-frame rotation are substituted. The
actual adapter, grasp scoring, intermediate-state reads, done/reward callbacks,
and observation assembly execute without importing Isaac Sim.
"""

import ast
import importlib.util
import sys
import torch
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab"
TASK = PACKAGE / "tasks/leap_hand_grasp"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


logic = load_module("grasp_env_contract_logic", TASK / "grasp_logic.py")
forces = load_module("grasp_env_contract_forces", TASK / "contact_forces.py")


def identity_rotation(quat, vector):
    """The simulator boundary supplies only identity xyzw orientations here."""
    torch.testing.assert_close(quat, torch.tensor([0.0, 0.0, 0.0, 1.0]).expand_as(quat))
    return vector


def native_env_class():
    parent_path = PACKAGE / "compat/inhand_direct/inhand_manipulation_env.py"
    parent_tree = ast.parse(parent_path.read_text())
    parent = next(node for node in parent_tree.body if isinstance(node, ast.ClassDef))
    parent.bases = []
    parent.body = [
        node
        for node in parent.body
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_intermediate_values"
    ]
    scope = {"torch": torch}
    exec(
        compile(ast.Module(body=[parent], type_ignores=[]), str(parent_path), "exec"),
        scope,
    )
    scope.update(
        LeapHandDirectEnv=scope[parent.name],
        read_contact_sample=forces.read_contact_sample,
        read_support_load=forces.read_support_load,
        score_grasp=logic.score_grasp,
        quat_apply=identity_rotation,
        quat_apply_inverse=identity_rotation,
    )
    path = TASK / "leap_grasp_env.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = {
        "fingertip_contact_forces",
        "_compute_intermediate_values",
        "_measure_grasp",
        "_get_dones",
        "_get_rewards",
        "_get_observations",
    }
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    return scope[cls.name]


def sensor(normal, friction, position):
    """Supply the backend's public one-body/one-partner sensor data contract."""
    return SimpleNamespace(
        data=SimpleNamespace(
            normal_force_matrix_w=normal[:, None, None],
            friction_force_matrix_w=friction[:, None, None],
            contact_pos_w=position[:, None, None],
            normal_force_matrix_w_history=normal[:, None, None, None].repeat(1, 4, 1, 1, 1),
            friction_force_matrix_w_history=friction[:, None, None, None].repeat(1, 4, 1, 1, 1),
        )
    )


def native_env():
    """Three environments: four contacts, one absent contact, all contacts absent."""
    env = native_env_class()()
    env.num_envs = 3
    env.physics_dt = 1 / 120
    env.step_dt = 1 / 30
    env.max_episode_length = 300
    env.num_fingertips = 4
    env.finger_bodies = env._tip_indices = list(range(4))
    env.cfg = SimpleNamespace(
        fingertip_body_names=["index", "thumb", "middle", "ring"],
        grasp=logic.GraspCriteria(),
        pulp_contact_tolerance=0.004,
        pulp_normal_alignment=0.4,
        cube_half_extent=0.03,
        hold_duration=2.0,
        force_observation_scale=10.0,
        fall_penalty=-5.0,
        unsafe_penalty=-5.0,
    )
    env.scene = SimpleNamespace(env_origins=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    env.in_hand_pos = torch.tensor([[0.0, 0.0, 0.6]]).repeat(3, 1)
    position = env.in_hand_pos + env.scene.env_origins
    rotation = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)
    tip_position = position[:, None].repeat(1, 4, 1)
    tip_rotation = rotation[:, None].repeat(1, 4, 1)

    def proxy(value):
        return SimpleNamespace(torch=value)

    env.hand = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=proxy(tip_position),
            body_link_pos_w=proxy(tip_position),
            body_quat_w=proxy(tip_rotation),
            body_link_quat_w=proxy(tip_rotation),
            body_vel_w=proxy(torch.zeros(3, 4, 6)),
            body_link_vel_w=proxy(torch.zeros(3, 4, 6)),
            joint_pos=proxy(torch.zeros(3, 16)),
            joint_vel=proxy(torch.zeros(3, 16)),
        )
    )
    env.object = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=proxy(position),
            root_link_pos_w=proxy(position),
            root_quat_w=proxy(rotation),
            root_vel_w=proxy(torch.zeros(3, 6)),
            root_link_vel_w=proxy(torch.zeros(3, 6)),
            root_lin_vel_w=proxy(torch.zeros(3, 3)),
            root_ang_vel_w=proxy(torch.zeros(3, 3)),
        )
    )
    normal = torch.tensor([2.0, 0.0, 0.0]).repeat(3, 4, 1)
    friction = torch.tensor([0.0, 0.2, 0.0]).repeat(3, 4, 1)
    contact_position = tip_position.clone()
    absent = torch.tensor(
        [
            [False, False, False, False],
            [True, False, False, False],
            [True, True, True, True],
        ]
    )
    normal[absent] = 0
    friction[absent] = 0
    contact_position[absent] = float("nan")
    env.fingertip_contact_sensors = {
        name: sensor(normal[:, i], friction[:, i], contact_position[:, i])
        for i, name in enumerate(env.cfg.fingertip_body_names)
    }
    env.support_sensor = sensor(torch.zeros(3, 3), torch.zeros(3, 3), torch.full((3, 3), float("nan")))
    env._pad_centers = torch.zeros(4, 3)
    env._pad_half_extents = torch.full((4, 3), 0.01)
    env._pad_normals = torch.tensor([[-1.0, 0.0, 0.0]]).repeat(4, 1)
    env.hand_dof_lower_limits = torch.full((3, 16), -2.0)
    env.hand_dof_upper_limits = torch.full((3, 16), 2.0)
    env.cur_targets = torch.zeros(3, 16)
    env.actions = torch.zeros(3, 16)
    env.target_delta = torch.zeros(3, 16)
    env._fresh_reset = torch.zeros(3, dtype=torch.bool)
    env.episode_length_buf = torch.ones(3, dtype=torch.long)
    env.grasp_state = logic.GraspEpisodeState(3, "cpu")
    env.successes = torch.zeros(3)
    env.extras = {}
    return env


class GraspEnvironmentContractTests(unittest.TestCase):
    def test_absent_contact_is_invalid_not_overload_or_nan_observation(self):
        env = native_env()
        terminated, truncated = env._get_dones()
        reward = env._get_rewards()
        obs = env._get_observations()["policy"]
        self.assertEqual(terminated.tolist(), [False, False, False])
        self.assertFalse(truncated.any())
        self.assertTrue(torch.isfinite(reward).all())
        self.assertTrue(torch.isfinite(obs).all())
        self.assertEqual(tuple(obs.shape), (3, 124))
        torch.testing.assert_close(obs[:, 123], torch.full((3,), 299 / 300))
        self.assertEqual(env.grasp_state.valid_steps.tolist(), [1, 0, 0])
        self.assertEqual(env.grasp_state.overload_steps.tolist(), [0, 0, 0])
        # Observation contract: full force, normal force, positions, validity, slip.
        torch.testing.assert_close(obs[0, 77:80], torch.tensor([0.2, 0.02, 0.0]))
        torch.testing.assert_close(obs[0, 89:92], torch.tensor([0.2, 0.0, 0.0]))
        torch.testing.assert_close(obs[2, 77:121], torch.zeros(44))
        self.assertEqual(obs[1, 113:117].tolist(), [0.0, 1.0, 1.0, 1.0])
        self.assertEqual(float(obs[1, 117]), 0.0)

    def test_pre_reset_terminal_observation_stays_finite_when_contacts_are_absent(self):
        for terminal_kind in ("timeout", "drop"):
            with self.subTest(terminal_kind=terminal_kind):
                env = native_env()
                if terminal_kind == "timeout":
                    env.episode_length_buf.fill_(env.max_episode_length)
                else:
                    env.object.data.root_link_pos_w.torch[:, 1] += 0.2
                terminated, truncated = env._get_dones()
                reward = env._get_rewards()
                # DirectRLEnv captures this callback before same-step autoreset;
                # checking only the returned post-reset observation misses NaNs.
                final_obs = env._get_observations()["policy"]
                self.assertEqual(terminated.tolist(), [terminal_kind == "drop"] * 3)
                self.assertEqual(truncated.tolist(), [terminal_kind == "timeout"] * 3)
                self.assertTrue(torch.isfinite(reward).all())
                self.assertTrue(torch.isfinite(final_obs).all())
                self.assertEqual(tuple(final_obs.shape), (3, 124))
                remaining = 0.0 if terminal_kind == "timeout" else 299 / 300
                torch.testing.assert_close(final_obs[:, 123], torch.full((3,), remaining))
                torch.testing.assert_close(final_obs[2, 77:121], torch.zeros(44))
                self.assertEqual(env.grasp_state.overload_steps.tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
