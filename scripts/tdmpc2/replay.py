"""Bounded, CPU-resident fixed-horizon replay for vector TD-MPC2 collectors."""

import torch


class SequenceReplay:
    """Store explicit pre-reset successor observations, never acting reset observations.

    Capacity is rounded down to a multiple of ``num_envs``. Sampling is with
    replacement and uses a replay-owned CPU generator, independent of policy RNG.
    """

    def __init__(
        self, capacity, num_envs, obs_dim, action_dim, horizon, batch_size, device="cpu", sample_device="cuda:0", seed=0
    ):
        for name, value in (
            ("capacity", capacity),
            ("num_envs", num_envs),
            ("obs_dim", obs_dim),
            ("action_dim", action_dim),
            ("horizon", horizon),
            ("batch_size", batch_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if torch.device(device).type != "cpu":
            raise ValueError("Replay storage device must be CPU; use sample_device for GPU batches")
        if capacity // num_envs < horizon:
            raise ValueError("capacity must hold at least horizon transitions per environment")
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.sample_device = torch.device(sample_device)
        self.slots = capacity // num_envs
        self.capacity = self.slots * num_envs
        shape = (self.slots, num_envs)
        self.obs = torch.empty((*shape, obs_dim), dtype=torch.float32, device=self.device)
        self.next_obs = torch.empty_like(self.obs)
        self.action = torch.empty((*shape, action_dim), dtype=torch.float32, device=self.device)
        self.reward = torch.empty((*shape, 1), dtype=torch.float32, device=self.device)
        self.terminated = torch.empty((*shape, 1), dtype=torch.float32, device=self.device)
        self.truncated = torch.zeros(shape, dtype=torch.bool)
        self._done = torch.zeros(shape, dtype=torch.bool)
        self._time = torch.full((self.slots,), -1, dtype=torch.int64)
        self._valid = torch.zeros(shape, dtype=torch.bool)
        self._eligible = None
        self._step = 0
        self._size = 0
        self._episode_length = torch.zeros(num_envs, dtype=torch.int64)
        self._short_episodes = 0
        self._short_episode_transitions = 0
        self._rng = torch.Generator(device="cpu").manual_seed(seed)
        self.condition_ids = torch.full(shape, -1, dtype=torch.int64)
        self._conditions = {}
        self._condition_lookup = {}
        self._condition_counts = {}
        self._next_condition_id = 0

    def __len__(self):
        return self._size

    def _starts(self):
        if self._eligible is None:
            self._eligible = self._valid.flatten().nonzero().flatten()
        return self._eligible

    @property
    def ready(self):
        """Whether at least one complete horizon can be sampled."""
        return self._starts().numel() > 0

    def can_sample(self):
        return self.ready

    @property
    def stats(self):
        """Replay occupancy and lifetime counts of unsampleable short episodes."""
        return {
            "transitions": len(self),
            "eligible_sequences": self._starts().numel(),
            "short_episodes": self._short_episodes,
            "short_episode_transitions": self._short_episode_transitions,
        }

    def add(self, obs, action, reward, next_obs, terminated, truncated, condition=None):
        """Append one vector step; ``next_obs`` must precede auto-reset.

        ``condition`` is an optional string stamp shared by this vector step,
        recorded as provenance only (not a model input or episode boundary).
        Rewards and flags may be ``(N,)`` or ``(N, 1)``; other inputs are ``(N, D)``.
        """
        if condition is not None and not isinstance(condition, str):
            raise ValueError("condition must be a string stamp or None")
        values = {}
        for name, value, width in (
            ("obs", obs, self.obs_dim),
            ("action", action, self.action_dim),
            ("reward", reward, 1),
            ("next_obs", next_obs, self.obs_dim),
            ("terminated", terminated, 1),
            ("truncated", truncated, 1),
        ):
            value = torch.as_tensor(value).detach()
            shapes = [(self.num_envs, width)]
            if name in ("reward", "terminated", "truncated"):
                shapes.append((self.num_envs,))
            if tuple(value.shape) not in shapes:
                raise ValueError(f"{name} must have shape {shapes}, received {tuple(value.shape)}")
            values[name] = value.to(device="cpu", dtype=torch.float32).reshape(self.num_envs, width)
        for name in ("terminated", "truncated"):
            if not ((values[name] == 0) | (values[name] == 1)).all():
                raise ValueError(f"{name} must contain only boolean or 0/1 flags")
        slot = self._step % self.slots
        for name in ("obs", "action", "reward", "next_obs", "terminated"):
            getattr(self, name)[slot].copy_(values[name])
        self.truncated[slot].copy_(values["truncated"].flatten())
        self._done[slot] = self.terminated[slot, :, 0].bool() | self.truncated[slot]
        old_id = int(self.condition_ids[slot, 0])
        if old_id >= 0:
            self._condition_counts[old_id] -= 1
            if self._condition_counts[old_id] == 0:
                del self._condition_counts[old_id]
                del self._condition_lookup[self._conditions.pop(old_id)]
        if condition not in self._condition_lookup:
            condition_id = self._next_condition_id
            self._next_condition_id += 1
            self._conditions[condition_id] = condition
            self._condition_lookup[condition] = condition_id
            self._condition_counts[condition_id] = 0
        condition_id = self._condition_lookup[condition]
        self.condition_ids[slot] = condition_id
        self._condition_counts[condition_id] += 1
        self._episode_length += 1
        short = self._done[slot] & (self._episode_length < self.horizon)
        self._short_episodes += int(short.sum())
        self._short_episode_transitions += int(self._episode_length[short].sum())
        self._episode_length[self._done[slot]] = 0
        self._time[slot] = self._step
        self._step += 1
        self._size = min(self.capacity, self._size + self.num_envs)
        # Only starts whose horizon contains the overwritten/new slot can change.
        starts = (slot - torch.arange(self.horizon)) % self.slots
        rows = (starts[:, None] + torch.arange(self.horizon)) % self.slots
        times = self._time[rows]
        contiguous = (times[:, :1] >= 0) & (times == times[:, :1] + torch.arange(self.horizon)).all(dim=1)[:, None]
        self._valid[starts] = contiguous & ~self._done[rows[:, :-1]].any(dim=1)
        self._eligible = None

    def state_dict(self):
        """Return complete CPU state, including pending continuity and sampler RNG.

        Large tensors are views, not copies: serialize synchronously between adds.
        Loading preserves collector continuity. Call ``start_new_stream()`` after
        load when the simulator was restarted instead of restored exactly.
        """
        buffers = {
            name: getattr(self, name)
            for name in (
                "obs",
                "next_obs",
                "action",
                "reward",
                "terminated",
                "truncated",
                "_done",
                "_time",
                "_valid",
                "_episode_length",
                "condition_ids",
            )
        }
        return {
            "version": 1,
            "config": {
                name: getattr(self, name)
                for name in ("capacity", "num_envs", "obs_dim", "action_dim", "horizon", "batch_size")
            },
            "buffers": buffers,
            "step": self._step,
            "size": self._size,
            "short_episodes": self._short_episodes,
            "short_episode_transitions": self._short_episode_transitions,
            "rng_state": self._rng.get_state(),
            "conditions": dict(self._conditions),
            "condition_counts": dict(self._condition_counts),
            "next_condition_id": self._next_condition_id,
        }

    def load_state_dict(self, state):
        """Restore complete replay into preallocated CPU storage."""
        if state.get("version") != 1:
            raise ValueError("Unsupported replay checkpoint version")
        for name in ("capacity", "num_envs", "obs_dim", "action_dim", "horizon", "batch_size"):
            if state.get("config", {}).get(name) != getattr(self, name):
                raise ValueError(f"Incompatible replay checkpoint {name}")
        expected = self.state_dict()["buffers"]
        buffers = state.get("buffers", {})
        if buffers.keys() != expected.keys():
            raise ValueError("Replay checkpoint buffers are incomplete or unknown")
        for name, value in buffers.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype
            ):
                raise ValueError(f"Incompatible replay checkpoint buffers: {name}")
        for name, value in buffers.items():
            getattr(self, name).copy_(value)
        self._step = state["step"]
        self._size = state["size"]
        self._short_episodes = state["short_episodes"]
        self._short_episode_transitions = state["short_episode_transitions"]
        self._rng.set_state(state["rng_state"].cpu())
        self._conditions = dict(state["conditions"])
        self._condition_lookup = {value: key for key, value in self._conditions.items()}
        self._condition_counts = dict(state["condition_counts"])
        self._next_condition_id = state["next_condition_id"]
        self._eligible = None

    def start_new_stream(self):
        """Cut collector continuity without discarding replay or changing terminal masks.

        Call after resetting/recreating the simulator, including checkpoint resume
        without simulator state. Already complete sequences remain sampleable.
        """
        if self._step:
            self._done[(self._step - 1) % self.slots] = True
        self._episode_length.zero_()

    def clear_pending(self):
        """Alias for :meth:`start_new_stream`; no complete transitions are removed."""
        self.start_new_stream()

    def sample(self, batch_size=None):
        """Return time-major ``(obs, action, reward, terminated, None)``."""
        starts = self._starts()
        if not starts.numel():
            raise RuntimeError("Replay has no complete sequence to sample")
        batch_size = self.batch_size if batch_size is None else batch_size
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        chosen = starts[torch.randint(starts.numel(), (batch_size,), generator=self._rng)]
        envs = chosen % self.num_envs
        rows = (chosen[None, :] // self.num_envs + torch.arange(self.horizon)[:, None]) % self.slots
        obs = torch.cat((self.obs[rows, envs], self.next_obs[rows[-1:], envs]), dim=0)
        return tuple(
            tensor.to(self.sample_device)
            for tensor in (obs, self.action[rows, envs], self.reward[rows, envs], self.terminated[rows, envs])
        ) + (None,)
