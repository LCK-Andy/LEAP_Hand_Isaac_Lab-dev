"""CPU-only exploration contract tests; load the module without task registration/Isaac."""

import importlib.util
import math
import torch
from pathlib import Path
from types import SimpleNamespace

import pytest
from rl_games.algos_torch.models import ModelA2CContinuousLogStd

MODULE = (
    Path(__file__).resolve().parents[1]
    / "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/dexsuite_reorient_pbt/pbt/exploration.py"
)
spec = importlib.util.spec_from_file_location("dexpbt_exploration", MODULE)
exploration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exploration)
ExplorationObserver = exploration.ExplorationObserver


class TinyNetwork(torch.nn.Module):
    def __init__(self, size=3):
        super().__init__()
        self.fixed_sigma = True
        self.sigma = torch.nn.Parameter(torch.zeros(size))
        self.sigma_act = torch.nn.Identity()
        self.mu = torch.nn.Parameter(torch.zeros(size))

    def forward(self, inputs):
        mean = self.mu.expand(inputs["obs"].shape[0], -1)
        return mean, self.sigma + torch.zeros_like(mean), mean[:, :1], None


class FakeAlgorithm:
    """The same observer/restore/action/optimizer seam used by rl-games 1.6.5."""

    def __init__(self, size=3):
        self.model = ModelA2CContinuousLogStd.Network(
            TinyNetwork(size),
            obs_shape=(1,),
            normalize_value=False,
            normalize_input=False,
            value_size=1,
        )
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=10.0)
        self.writer = None
        self.frame = 0

    def get_action_values(self, obs):
        with torch.no_grad():
            return self.model({"obs": obs, "is_train": False})

    def restore(self, logstd, set_epoch=True):
        with torch.no_grad():
            self.model.a2c_network.sigma.copy_(logstd)


def test_restored_logstd_is_bounded_before_first_rollout_and_each_optimizer_step():
    algo = FakeAlgorithm()
    observer = ExplorationObserver()
    observer.after_init(algo)
    sigma = algo.model.a2c_network.sigma
    torch.testing.assert_close(sigma.exp(), torch.full((3,), 0.3))

    # after_init precedes Runner._restore; keep in-range donor exploration, not .3.
    algo.restore(torch.tensor([math.log(0.2), math.log(50.0), math.log(0.001)]))
    torch.testing.assert_close(sigma.exp(), torch.tensor([0.2, 1.0, 0.05]))
    first = algo.get_action_values(torch.zeros(8, 1))
    torch.testing.assert_close(first["sigmas"][0], torch.tensor([0.2, 1.0, 0.05]))

    # No intervening rollout: a rollout-only clamp would fail on the first step.
    for direction, expected in [(-1.0, 1.0), (1.0, 0.05), (-1.0, 1.0)]:
        algo.optimizer.zero_grad()
        (direction * sigma.sum()).backward()
        algo.optimizer.step()
        torch.testing.assert_close(sigma.exp(), torch.full((3,), expected))
    observer.close()


def test_actual_sample_diagnostics_are_weighted_and_do_not_modify_outputs():
    algo = FakeAlgorithm()
    outputs = [
        {
            "actions": torch.tensor([[1.5, 0.0, -2.0]]),
            "mus": torch.tensor([[0.0, 2.0, 0.0]]),
        },
        {"actions": torch.zeros(3, 3), "mus": torch.zeros(3, 3)},
    ]
    algo.get_action_values = lambda obs: outputs.pop(0)
    observer = ExplorationObserver()
    observer.after_init(algo)
    result = algo.get_action_values(None)
    torch.testing.assert_close(result["actions"], torch.tensor([[1.5, 0.0, -2.0]]))
    algo.get_action_values(None)
    metrics = observer.metrics()
    assert metrics["exploration/global/sampled_clip_fraction"] == 2 / 12
    assert metrics["exploration/global/mean_saturation_fraction"] == 1 / 12
    assert metrics["exploration/global/sampled_coordinates"] == 12
    assert abs(metrics["exploration/global/std_mean"] - 0.3) < 1e-6
    observer.close()


def test_runner_sigma_override_is_projected_and_gaussian_logprob_remains_correct():
    algo = FakeAlgorithm()
    observer = ExplorationObserver()
    observer.after_init(algo)
    # Runner's sigma override writes the raw *logstd* after restoring.
    with torch.no_grad():
        algo.model.a2c_network.sigma.fill_(4.0)
    obs = torch.zeros(32, 1)
    result = algo.get_action_values(obs)
    torch.testing.assert_close(result["sigmas"], torch.ones(32, 3))
    expected = -torch.distributions.Normal(result["mus"], result["sigmas"]).log_prob(result["actions"]).sum(-1)
    torch.testing.assert_close(result["neglogpacs"], expected)
    training = algo.model({"obs": obs, "is_train": True, "prev_actions": result["actions"]})
    torch.testing.assert_close(training["prev_neglogp"], result["neglogpacs"])
    training["prev_neglogp"].mean().backward()
    assert torch.isfinite(algo.model.a2c_network.sigma.grad).all()
    algo.optimizer.step()
    assert algo.model.a2c_network.sigma.exp().max() <= 1
    observer.close()


def test_scoped_hooks_are_reversible_and_other_algorithm_is_untouched():
    algo, other = FakeAlgorithm(), FakeAlgorithm()
    observer = ExplorationObserver()
    original_action = algo.get_action_values
    original_restore = algo.restore
    observer.after_init(algo)
    assert "get_action_values" not in other.__dict__
    torch.testing.assert_close(other.model.a2c_network.sigma.exp(), torch.ones(3))
    observer.close()
    observer.close()
    assert algo.get_action_values == original_action
    assert algo.restore == original_restore
    algo.restore(torch.full((3,), 4.0))
    algo.optimizer.zero_grad()
    (-algo.model.a2c_network.sigma.sum()).backward()
    algo.optimizer.step()
    assert (algo.model.a2c_network.sigma > 4.0).all()


def test_exact_action_order_gets_group_profile_without_assuming_arm_first():
    names = [f"{finger}_joint_{i}" for finger in ("thumb", "index", "middle", "ring") for i in range(4)]
    names += [f"iiwa7_joint_{i}" for i in range(7, 0, -1)]
    algo = FakeAlgorithm(23)
    observer = ExplorationObserver(action_joint_names=names)
    observer.after_init(algo)
    torch.testing.assert_close(algo.model.a2c_network.sigma.exp()[:16], torch.full((16,), 0.3))
    torch.testing.assert_close(algo.model.a2c_network.sigma.exp()[16:], torch.full((7,), 0.15))
    algo.restore(torch.full((23,), 4.0))
    result = algo.get_action_values(torch.zeros(4, 1))
    torch.testing.assert_close(result["sigmas"][0, :16], torch.ones(16))
    torch.testing.assert_close(result["sigmas"][0, 16:], torch.full((7,), 0.5))
    metrics = observer.metrics()
    assert metrics["exploration/arm/std_max"] == 0.5
    assert metrics["exploration/fingers/std_max"] == 1.0
    observer.close()


@pytest.mark.parametrize("names", [None, ["unknown"] * 23, ["iiwa7_joint_1"] * 23, ["iiwa7_joint_1"]])
def test_unknown_mapping_uses_uniform_profile_and_global_only(names):
    algo = FakeAlgorithm(23)
    observer = ExplorationObserver(action_joint_names=names)
    observer.after_init(algo)
    torch.testing.assert_close(algo.model.a2c_network.sigma.exp(), torch.full((23,), 0.3))
    assert not any("/arm/" in key or "/fingers/" in key for key in observer.metrics())
    observer.close()


def test_stats_writer_resets_only_sample_window_and_empty_window_is_not_measurement():
    algo = FakeAlgorithm()
    logged = {}
    algo.writer = SimpleNamespace(add_scalar=lambda key, value, frame: logged.update({key: (value, frame)}))
    observer = ExplorationObserver()
    observer.after_init(algo)
    assert "exploration/global/sampled_clip_fraction" not in observer.metrics()
    algo.get_action_values(torch.zeros(4, 1))
    observer.after_print_stats(100, 1, 2.0)
    assert logged["exploration/global/sampled_coordinates"] == (12, 100)
    assert "exploration/global/sampled_clip_fraction" in logged
    assert "exploration/global/sampled_clip_fraction" not in observer.metrics()
    observer.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_restored_sigma_is_repaired_and_reported(value, caplog):
    algo = FakeAlgorithm()
    observer = ExplorationObserver()
    observer.after_init(algo)
    algo.restore(torch.full((3,), value))
    result = algo.get_action_values(torch.zeros(2, 1))
    assert torch.isfinite(result["neglogpacs"]).all()
    assert observer.metrics()["exploration/nonfinite_logstd_coordinates_total"] == 3
    assert "projected 3/3" in caplog.text
    observer.close()


@pytest.mark.parametrize("kwargs", [{"min_std": 0}, {"max_std": 0.1}, {"initial_std": float("nan")}])
def test_invalid_profile_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ExplorationObserver(**kwargs)


@pytest.mark.parametrize("fused", [False, True])
def test_adam_optimizer_restore_keeps_poststep_projection(fused):
    algo = FakeAlgorithm()
    algo.optimizer = torch.optim.Adam(algo.model.parameters(), lr=10.0, fused=fused)
    observer = ExplorationObserver()
    observer.after_init(algo)
    sigma = algo.model.a2c_network.sigma
    for _ in range(3):
        algo.optimizer.zero_grad()
        (-sigma.sum()).backward()
        algo.optimizer.step()
        torch.testing.assert_close(sigma.exp(), torch.ones(3))
    state = algo.optimizer.state_dict()
    algo.optimizer.load_state_dict(state)
    algo.restore(torch.full((3,), math.log(0.2)))
    algo.optimizer.zero_grad()
    (-sigma.sum()).backward()
    algo.optimizer.step()
    torch.testing.assert_close(sigma.exp(), torch.ones(3))
    observer.close()


def test_wrong_gaussian_model_is_rejected_before_initialization():
    algo = FakeAlgorithm()
    algo.model = torch.nn.Linear(1, 3)
    observer = ExplorationObserver()
    with pytest.raises(TypeError, match="continuous_a2c_logstd"):
        observer.after_init(algo)
    assert "get_action_values" not in algo.__dict__


def test_wrong_gaussian_parameterization_is_rejected_without_patching():
    algo = FakeAlgorithm()
    algo.model.a2c_network.sigma_act = torch.nn.Softplus()
    observer = ExplorationObserver()
    with pytest.raises(TypeError, match="identity sigma"):
        observer.after_init(algo)
    assert "get_action_values" not in algo.__dict__
    observer.close()
