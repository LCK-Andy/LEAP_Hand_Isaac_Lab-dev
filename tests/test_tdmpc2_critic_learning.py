"""Critic learning must survive the functional ensemble and detached-policy paths."""

import torch

import pytest

from scripts.tdmpc2.agent import Agent
from scripts.tdmpc2.config import TDMPC2Config


def config():
    return TDMPC2Config(
        obs_dim=7,
        action_dim=3,
        num_envs=1,
        plan_chunk_size=1,
        batch_size=8,
        enc_dim=16,
        mlp_dim=16,
        latent_dim=16,
        num_q=2,
        num_samples=8,
        num_elites=3,
        num_pi_trajs=2,
        iterations=1,
        dropout=0.0,
    )


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_real_updates_learn_nonzero_return_in_every_critic(device):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    cfg = config()
    agent = Agent(cfg, device=device)
    obs = torch.randn(cfg.horizon + 1, cfg.batch_size, cfg.obs_dim, device=device)
    actions = torch.zeros(cfg.horizon, cfg.batch_size, cfg.action_dim, device=device)
    rewards = torch.full((cfg.horizon, cfg.batch_size, 1), 2.0, device=device)
    terminal = torch.zeros_like(rewards)
    terminal[-1] = 1.0

    class Replay:
        def sample(self):
            return obs, actions, rewards, terminal, None

    before = {name: tensor.detach().clone() for name, tensor in agent.model._Qs.named_parameters()}
    initial = agent.update(Replay())
    for _ in range(99):
        latest = agent.update(Replay())
    for name, parameter in agent.model._Qs.named_parameters():
        changed_per_critic = (parameter.detach() - before[name]).reshape(cfg.num_q, -1).abs().amax(dim=1)
        assert bool((changed_per_critic > 0).all()), f"Frozen critic parameter: {name}"
        assert parameter in agent.core.optim.state, f"Missing critic Adam state: {name}"
    assert float(latest["value_loss"]) < float(initial["value_loss"]) - 0.01
    with torch.no_grad():
        z = agent.model.encode(obs[0], None)
        q = agent.model.Q(z, actions[0], None, return_type="avg")
    assert float(q.mean()) > 0.01, "Known positive returns should not remain uniform/zero-valued"


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_detached_policy_path_keeps_action_gradients_but_not_critic_gradients(device):
    torch.manual_seed(17)
    agent = Agent(config(), device=device)
    with torch.no_grad():
        agent.model._Qs.params["2", "weight"].normal_(0, 0.2)
    z = torch.randn(8, 16, device=device)
    action = torch.randn(8, 3, device=device, requires_grad=True)
    value = agent.model.Q(z, action, None, return_type="avg", detach=True)
    value.sum().backward()
    assert action.grad is not None and bool((action.grad.abs() > 0).any())
    assert all(p.grad is None for p in agent.model._Qs.parameters())
    agent.core.pi_optim.zero_grad(set_to_none=True)
    before = [p.detach().clone() for p in agent.model._pi.parameters()]
    agent.core.update_pi(torch.randn(4, 8, 16, device=device), None)
    assert any(not torch.equal(p.detach(), old) for p, old in zip(agent.model._pi.parameters(), before))
    assert all(p.grad is None for p in agent.model._Qs.parameters())


def test_target_is_independent_and_tracks_trained_critics():
    agent = Agent(config(), device="cpu")
    online = agent.model._Qs.params["2", "weight"]
    target = agent.model._target_Qs_params["2", "weight"]
    before = target.detach().clone()
    with torch.no_grad():
        online.add_(2.0)
    torch.testing.assert_close(target, before, rtol=0, atol=0)
    agent.model.soft_update_target_Q()
    torch.testing.assert_close(target, before + 0.02, rtol=1e-6, atol=1e-7)
    agent.model.train()
    assert agent.model._Qs.module.training
    assert agent.model._detach_Qs.module.training
    assert not agent.model._target_Qs.module.training


def test_agent_reports_all_critic_optimizer_states_and_refuses_disconnected_updates():
    agent = Agent(config(), device="cpu")
    diagnostics = agent.critic_diagnostics()
    assert diagnostics["critic/parameter_tensors"] == 10
    assert diagnostics["critic/optimizer_tensors"] == 0

    class Replay:
        def sample(self):
            return torch.randn(4, 8, 7), torch.zeros(3, 8, 3), torch.ones(3, 8, 1), torch.zeros(3, 8, 1), None

    # Reproduce a disconnected Q path without changing the loss or optimizer membership.
    original = agent.model._Qs.forward
    agent.model._Qs.forward = lambda x: original(x).detach().requires_grad_()
    with pytest.raises(RuntimeError, match="critic.*optimizer"):
        agent.update(Replay())
