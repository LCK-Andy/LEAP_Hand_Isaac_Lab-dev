"""Released DexPBT tensor contract, independent of simulator imports."""

import ast
import hashlib
import importlib.util
import numpy as np
import sys
import torch
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexpbt_original/mdp.py"
spec = importlib.util.spec_from_file_location("dexpbt_original_mdp", PATH)
mdp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mdp
spec.loader.exec_module(mdp)


class ReleasedControlTests(unittest.TestCase):
    def test_arm_is_relative_but_fingers_are_absolute(self):
        action = torch.zeros(1, 23)
        action[:, 0] = 1
        action[:, 7] = -1
        lower, upper = torch.full((23,), -2.0), torch.full((23,), 4.0)
        previous = torch.full((1, 23), 0.5)
        targets = mdp.joint_targets(action, previous, lower, upper)
        self.assertAlmostEqual(targets[0, 0].item(), 0.6667, places=6)
        torch.testing.assert_close(targets[0, 1:7], torch.full((6,), 0.5))
        self.assertEqual(targets[0, 7].item(), -2.0)
        torch.testing.assert_close(targets[0, 8:], torch.ones(15))
        torch.testing.assert_close(previous, torch.full((1, 23), 0.5))


FIXTURE = ROOT / "tests/fixtures/dexpbt_original/isaacgymenvs/tasks/allegro_kuka"
BASE_HASH = "367587172eb2d78834d20dbd0cf701322c8378368f48f9679765e3697aab331b"


def original_math():
    path = FIXTURE.parents[1] / "utils/torch_jit_utils.py"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "95602e6ef57c148143f682640e063eb27ded035d04a29fe5959de032177e6ca3"
    selected = [
        n
        for n in ast.parse(raw).body
        if isinstance(n, ast.FunctionDef) and n.name in {"quat_rotate", "scale", "unscale", "tensor_clamp"}
    ]
    for node in selected:
        node.decorator_list = []
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


def original_methods():
    """Compile only original pure methods, never execute upstream module imports."""
    path = FIXTURE / "allegro_kuka_base.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_HASH:
        raise AssertionError("Pinned original oracle source bytes changed")
    tree = ast.parse(raw)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AllegroKukaBase")
    names = {
        "_distance_delta_rewards",
        "_lifting_reward",
        "_keypoint_reward",
        "_action_penalties",
        "_compute_resets",
        "compute_kuka_reward",
        "compute_full_state",
        "clamp_obs",
        "compute_observations",
        "pre_physics_step",
        "get_random_quat",
    }
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    wrapper = ast.ClassDef(name="Original", bases=[], keywords=[], body=selected, decorator_list=[])
    namespace = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "Tuple": tuple,
        "np": np,
        "gymtorch": SimpleNamespace(unwrap_tensor=lambda tensor: tensor),
        "torch_rand_float": lambda lower, upper, shape, device: (upper - lower) * torch.rand(shape, device=device)
        + lower,
        **original_math(),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["Original"]


def fixture(n=4):
    state = mdp.make_state(n)
    state.closest_fingertip_dist.fill_(0.4)
    state.furthest_hand_dist.fill_(0.4)
    state.closest_keypoint_max_dist.fill_(0.8)
    state.progress_buf.fill_(598)
    distance = torch.full((n, 4), 0.3)
    geometry = mdp.Geometry(
        torch.zeros(n, 3),
        torch.zeros(n, 4, 3),
        distance,
        torch.zeros(n, 4, 3),
        torch.zeros(n, 4, 3),
        torch.full((n,), 0.6),
    )
    pos = torch.zeros(n, 3)
    pos[:, 2] = 0.65
    return state, geometry, pos, torch.zeros(n, 23)


def oracle(state, geometry, pos, vel, cfg):
    obj = original_methods()()
    for key, value in vars(state).items():
        setattr(obj, key, value.clone())
    for key, value in vars(geometry).items():
        setattr(obj, key, value.clone())
    obj.object_pos = pos.clone()
    obj.object_init_state = torch.zeros(len(pos), 13)
    obj.object_init_state[:, 2] = 0.63
    obj.arm_hand_dof_vel = vel.clone()
    for source_name, field_name in mdp.SOURCE_REWARD_FIELDS.items():
        attribute = {
            "distanceDeltaRewScale": "distance_delta_rew_scale",
            "liftingRewScale": "lifting_rew_scale",
            "liftingBonus": "lifting_bonus",
            "liftingBonusThreshold": "lifting_bonus_threshold",
            "keypointRewScale": "keypoint_rew_scale",
            "reachGoalBonus": "reach_goal_bonus",
            "kukaActionsPenaltyScale": "kuka_actions_penalty_scale",
            "allegroActionsPenaltyScale": "allegro_actions_penalty_scale",
            "successTolerance": "success_tolerance",
            "successSteps": "success_steps",
            "keypointScale": "keypoint_scale",
            "maxConsecutiveSuccesses": "max_consecutive_successes",
            "episodeLength": "max_episode_length",
            "fallDistance": "fall_dist",
            "fallPenalty": "fall_penalty",
        }[source_name]
        setattr(obj, attribute, getattr(cfg, field_name))
    obj.num_hand_arm_dofs, obj.num_allegro_fingertips = 23, 4
    obj.finger_rew_coeffs = torch.ones_like(geometry.curr_fingertip_distances)
    obj.reset_buf = torch.zeros(len(pos), dtype=torch.long)
    obj.reset_goal_buf = torch.zeros(len(pos), dtype=torch.long)
    obj.rew_buf = torch.zeros(len(pos))
    obj.prev_episode_successes = torch.zeros(len(pos))
    obj.extras = {}
    keys = [
        "raw_fingertip_delta_rew",
        "raw_hand_delta_penalty",
        "raw_lifting_rew",
        "raw_keypoint_rew",
        "fingertip_delta_rew",
        "hand_delta_penalty",
        "lifting_rew",
        "lift_bonus_rew",
        "keypoint_rew",
        "bonus_rew",
        "kuka_actions_penalty",
        "allegro_actions_penalty",
    ]
    obj.rewards_episode = {k: torch.zeros(len(pos)) for k in keys}
    obj._true_objective = lambda: torch.zeros(len(pos))
    obj._extra_reset_rules = lambda resets: torch.where(
        obj.curr_fingertip_distances.max(dim=-1).values > 1.5, torch.ones_like(resets), resets
    )
    return obj


class ReleasedRewardTests(unittest.TestCase):
    def test_multistep_reward_and_memory_match_unmodified_original(self):
        state, geometry, pos, vel = fixture()
        cfg = mdp.RewardConfig(success_steps=2)
        reference = oracle(state, geometry, pos, vel, cfg)
        for i in range(4):
            state.progress_buf += 1
            reference.progress_buf += 1
            pos[:, 2] = torch.tensor([0.65, 0.8, 0.05, 0.65])
            geometry.keypoints_max_dist[:] = 0.05 if i in (0, 2) else 0.5
            geometry.curr_fingertip_distances[3] = 1.6
            vel[:] = i * 0.1
            reference.object_pos[:] = pos
            reference.keypoints_max_dist[:] = geometry.keypoints_max_dist
            reference.curr_fingertip_distances[:] = geometry.curr_fingertip_distances
            reference.arm_hand_dof_vel[:] = vel
            expected_reward, expected_success = reference.compute_kuka_reward()
            actual = mdp.step_rewards(state, geometry, pos, torch.full((4,), 0.63), vel, cfg)
            torch.testing.assert_close(actual.reward, expected_reward)
            torch.testing.assert_close(actual.is_success, expected_success)
            torch.testing.assert_close(actual.reset, reference.reset_buf.bool())
            for name in vars(state):
                torch.testing.assert_close(getattr(state, name), getattr(reference, name))
            for name, value in reference.extras["episode_cumulative"].items():
                torch.testing.assert_close(actual.terms[name], value)
            # Runtime resets after emitted terminal transition; pure oracle mirrors this boundary.
            reference.reset_buf.zero_()

    def test_source_lift_and_success_have_no_contact_requirement(self):
        state, geometry, pos, vel = fixture(1)
        geometry.keypoints_max_dist.zero_()
        result = mdp.step_rewards(state, geometry, pos, torch.tensor([0.63]), vel, mdp.RewardConfig())
        self.assertTrue(result.is_success.item())
        self.assertFalse(state.lifted_object.item())
        self.assertAlmostEqual(result.reward.item(), 1021.4, places=3)
        self.assertEqual(state.progress_buf.item(), 0)


class ReleasedObservationTests(unittest.TestCase):
    def test_117_channel_order_and_pre_reward_memory_match_original(self):
        state, geometry, pos, vel = fixture(1)
        cfg = mdp.RewardConfig()
        reference = oracle(state, geometry, pos, vel, cfg)
        reference.num_envs, reference.full_state_size = 1, 117
        reference.with_dof_force_sensors = False
        reference.num_keypoints = 4
        lower, upper = -torch.ones(23), 2 * torch.ones(23)
        q = torch.arange(23).reshape(1, 23) / 23
        palm = torch.arange(13).reshape(1, 13) / 13
        obj = torch.arange(13).reshape(1, 13) / 26
        scales = torch.tensor([[1.0, 2.0, 0.75]])
        reference.arm_hand_dof_pos = q
        reference.arm_hand_dof_lower_limits, reference.arm_hand_dof_upper_limits = lower, upper
        reference._palm_state, reference.object_state, reference.object_scales = palm, obj, scales
        expected = torch.zeros(1, 117)
        count, reward_offset = reference.compute_full_state(expected)
        self.assertEqual((count, reward_offset), (117, 116))
        actual = mdp.pack_observations(q, vel, lower, upper, palm, obj, scales, geometry, state)
        torch.testing.assert_close(actual, expected)
        geometry.keypoints_max_dist.zero_()
        result = mdp.step_rewards(state, geometry, pos, torch.tensor([0.63]), vel, cfg)
        # Source encodes old success/progress/minimum/lift state, then current scaled reward.
        self.assertEqual(actual[0, 115].item(), 0.0)
        self.assertEqual(state.successes.item(), 1)
        self.assertGreater(actual[0, 114].item(), 4)
        output = mdp.finish_observations(actual, result.reward)
        self.assertEqual(output[0, 116].item(), 10.0)
        self.assertTrue(torch.equal(output, actual))

    def test_geometry_offsets_and_scaled_keypoints_are_in_environment_axes(self):
        obj, palm = torch.zeros(1, 13), torch.zeros(1, 13)
        tips = torch.zeros(1, 4, 13)
        obj[:, 6] = palm[:, 6] = tips[:, :, 6] = 1
        goal = obj[:, :7].clone()
        goal[:, 0] = 0.2
        geometry = mdp.compute_geometry(obj, palm, tips, goal, torch.ones(1, 3))
        torch.testing.assert_close(geometry.palm_center_pos, torch.tensor([[0.0, -0.02, 0.16]]))
        torch.testing.assert_close(geometry.fingertip_pos_rel_palm[0, 0], torch.tensor([0.05, 0.025, -0.16]))
        torch.testing.assert_close(geometry.keypoints_rel_goal, torch.tensor([[[-0.2, 0.0, 0.0]]]).repeat(1, 4, 1))
        torch.testing.assert_close(geometry.keypoints_max_dist, torch.tensor([0.2]))
        torch.testing.assert_close(geometry.keypoints_rel_palm[0, 0], torch.tensor([0.0375, 0.0575, -0.1225]))
        state = mdp.make_state(1)
        mdp.initialize_distances(state, geometry)
        torch.testing.assert_close(state.closest_keypoint_max_dist, torch.tensor([0.2]))
        mdp.reset_state(state, torch.tensor([0]), goal_only=True)
        self.assertEqual(state.closest_keypoint_max_dist.item(), -1)
        self.assertTrue((state.closest_fingertip_dist > 0).all())
        mdp.reset_state(state, torch.tensor([0]))
        self.assertTrue((state.closest_fingertip_dist == -1).all())


class ReleasedCurriculumTests(unittest.TestCase):
    def test_curriculum_and_objective_match_original_at_thresholds(self):
        path = FIXTURE / "allegro_kuka_utils.py"
        raw = path.read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), "a363db1eedd040f93c16b60381fea58833fc8b3c2f3493db2e33d7486217e257"
        )
        tree = ast.parse(raw)
        selected = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in {"tolerance_curriculum", "interp_0_1", "tolerance_successes_objective"}
        ]
        namespace = {"Tensor": torch.Tensor, "Tuple": tuple, "print": lambda *args: None}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
        for frames in (2999, 3000, 3001):
            for successes in (2.99, 3.0, 3.01):
                args = (0, frames, 3000, torch.tensor([successes]), 0.075, 0.075, 0.01, 0.9)
                self.assertEqual(mdp.tolerance_curriculum(*args), namespace["tolerance_curriculum"](*args))
        for tolerance in (0.075, 0.02, 0.01):
            args = (tolerance, 0.075, 0.01, torch.tensor([0.0, 3.0, 50.0]))
            torch.testing.assert_close(
                mdp.tolerance_successes_objective(*args), namespace["tolerance_successes_objective"](*args)
            )
        # Exactly three successes triggers in source, unlike prose Nsucc > 3.
        self.assertEqual(
            mdp.tolerance_curriculum(0, 3000, 3000, torch.tensor([3.0]), 0.075, 0.075, 0.01, 0.9), (0.0675, 3000)
        )

    def test_source_knobs_are_strict_and_fall_distance_remains_dead(self):
        state, geometry, pos, vel = fixture(1)
        cfg = mdp.RewardConfig.from_mapping(
            {"fallDistance": 999.0, "fallPenalty": -999.0, "distanceDeltaRewScale": 25.0}
        )
        self.assertEqual(cfg.to_source_mapping()["fallDistance"], 999.0)
        pos[:, 2] = 0.11
        result = mdp.step_rewards(state, geometry, pos, torch.tensor([0.63]), vel, cfg)
        self.assertFalse(result.reset.item())
        self.assertAlmostEqual(result.reward.item(), 10.0, places=5)
        for absent in ("distRewardScale", "rotRewardScale", "actionPenaltyScale"):
            with self.assertRaises(ValueError):
                mdp.RewardConfig.from_mapping({absent: 1.0})


class ReleasedOriginalGeometryTests(unittest.TestCase):
    def test_arbitrary_rotations_and_geometry_match_original_observation_method(self):
        generator = torch.Generator().manual_seed(791)
        n = 19
        original = original_methods()()
        original.gym = SimpleNamespace(
            **{
                name: lambda sim: None
                for name in (
                    "refresh_dof_state_tensor",
                    "refresh_actor_root_state_tensor",
                    "refresh_rigid_body_state_tensor",
                )
            }
        )
        original.sim, original.device, original.obs_type = None, "cpu", "full_state"
        original.with_fingertip_force_sensors = original.with_dof_force_sensors = False
        original.num_envs, original.num_keypoints, original.num_allegro_fingertips = n, 4, 4
        original.allegro_palm_handle, original.allegro_fingertip_handles = 0, [1, 2, 3, 4]
        original.root_state_tensor = torch.randn(n, 13, generator=generator)
        original.rigid_body_states = torch.randn(n, 5, 13, generator=generator)
        original.goal_states = torch.randn(n, 13, generator=generator)
        for values in (original.root_state_tensor, original.rigid_body_states, original.goal_states):
            values[..., 3:7] /= values[..., 3:7].norm(dim=-1, keepdim=True)
        original.object_indices = torch.arange(n)
        original.object_scales = torch.rand(n, 3, generator=generator) * 4 + 0.5
        original.palm_offset = np.array([0.0, -0.02, 0.16], dtype=np.float32)
        original.fingertip_offsets = np.array([[0.05, 0.005, 0.0]] * 3 + [[0.06, 0.005, 0.0]], dtype=np.float32)
        signs = ((1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1))
        # Original Python scalar offset calculation is rounded to float32 only AFTER construction.
        offsets = [
            [[sign[j] * float(sc[j]) * 0.05 * 1.5 / 2 for j in range(3)] for sign in signs]
            for sc in original.object_scales
        ]
        original.object_keypoint_offsets = torch.tensor(offsets)
        state = mdp.make_state(n)
        for name, value in vars(state).items():
            setattr(original, name, value.clone())
        original.obj_keypoint_pos = torch.zeros(n, 4, 3)
        original.goal_keypoint_pos = torch.zeros(n, 4, 3)
        original.obs_buf, original.full_state_size = torch.zeros(n, 117), 117
        original.compute_full_state = lambda buf: (117, 116)
        original.compute_observations()
        actual = mdp.compute_geometry(
            original.root_state_tensor,
            original.rigid_body_states[:, 0],
            original.rigid_body_states[:, 1:],
            original.goal_states[:, :7],
            original.object_scales,
        )
        for name in vars(actual):
            torch.testing.assert_close(getattr(actual, name), getattr(original, name), atol=1e-6, rtol=1e-6)
        mdp.initialize_distances(state, actual)
        for name in ("closest_fingertip_dist", "furthest_hand_dist", "closest_keypoint_max_dist"):
            torch.testing.assert_close(getattr(state, name), getattr(original, name), atol=1e-6, rtol=1e-6)

    def test_multistep_control_matches_original_pre_physics_step(self):
        original = original_methods()()
        original.device, original.sim = "cpu", None
        original.privileged_actions, original.use_relative_control = False, False
        original.force_scale, original.num_hand_arm_dofs = 0.0, 23
        original.reset_buf = original.reset_goal_buf = torch.zeros(11, dtype=torch.long)
        original.reset_target_pose = lambda ids: None
        original.set_actor_root_state_tensor_indexed = lambda: None
        original.gym = SimpleNamespace(set_dof_position_target_tensor=lambda sim, tensor: None)
        original.arm_hand_dof_lower_limits = torch.linspace(-3.0, -0.1, 23)
        original.arm_hand_dof_upper_limits = torch.linspace(0.2, 4.0, 23)
        original.dt, original.hand_dof_speed_scale, original.act_moving_average = 0.01667, 10.0, 0.7
        original.prev_targets = torch.zeros(11, 23)
        original.cur_targets = torch.zeros(11, 23)
        previous = original.prev_targets.clone()
        generator = torch.Generator().manual_seed(453)
        for _ in range(8):
            actions = torch.randn(11, 23, generator=generator) * 2
            original.pre_physics_step(actions)
            previous = mdp.joint_targets(
                actions,
                previous,
                original.arm_hand_dof_lower_limits,
                original.arm_hand_dof_upper_limits,
                moving_average=0.7,
            )
            torch.testing.assert_close(previous, original.cur_targets, atol=0.0, rtol=0.0)

    def test_timeout_success_and_hold_accumulation_edge_cases(self):
        state, geometry, pos, vel = fixture(2)
        state.progress_buf[:] = 599
        state.successes[0] = 49
        geometry.keypoints_max_dist[:] = torch.tensor([0.01, 0.5])
        result = mdp.step_rewards(state, geometry, pos, torch.full((2,), 0.63), vel, mdp.RewardConfig())
        self.assertTrue(result.reset.all())
        self.assertEqual(result.timeout.tolist(), [False, True])
        self.assertEqual(state.progress_buf.tolist(), [0, 599])
        mdp.reset_state(state, torch.arange(2))
        mdp.initialize_distances(state, geometry)
        for distance in (0.01, 0.5, 0.01):
            geometry.keypoints_max_dist[:] = distance
            result = mdp.step_rewards(
                state, geometry, pos, torch.full((2,), 0.63), vel, mdp.RewardConfig(success_steps=2)
            )
        self.assertTrue(result.is_success.all())  # no reset of accumulated near-goal steps on the miss


if __name__ == "__main__":
    unittest.main()
