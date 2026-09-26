"""Strict full-force contact samples without importing a simulator backend.

Forces are in newtons, in world axes, ON the sensing body (not on its partner).
Only explicit normal/friction matrices are used: rc1's legacy ``force_matrix_w``
alias is normal-only on PhysX/OVPhysX. Histories retain newest-first ordering and
are snapshots of sensor updates, not necessarily solver substeps. The environment
must arrange a sensor update on every required physics tick.
"""

import math
import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class ContactSample:
    """One body/partner sample; vectors are (N,3), histories (N,H,3)."""

    normal: torch.Tensor
    friction: torch.Tensor
    total: torch.Tensor
    position: torch.Tensor
    position_valid: torch.Tensor
    history_normal: torch.Tensor
    history_friction: torch.Tensor


def _tensor(value, name: str) -> torch.Tensor:
    if value is None:
        raise RuntimeError(f"Required contact data '{name}' is unavailable; no reduced-force fallback is allowed.")
    if not isinstance(value, torch.Tensor):
        value = value.torch if hasattr(value, "torch") else torch.from_dlpack(value)
    return value


def _required(data, name: str) -> torch.Tensor:
    try:
        return _tensor(getattr(data, name), name)
    except (AttributeError, NotImplementedError, TypeError) as exc:
        raise RuntimeError(
            f"Required contact data '{name}' is unavailable on {type(data).__module__}; "
            "enable object filtering and friction/history/point tracking. No reduced-force fallback is allowed."
        ) from exc


def _validate(value: torch.Tensor, shape: tuple[int, ...], name: str, *, finite: bool = True) -> None:
    if value.shape != shape:
        raise RuntimeError(f"Contact '{name}' shape {tuple(value.shape)} does not match required {shape}.")
    if not value.is_floating_point() or (finite and not torch.isfinite(value).all()):
        raise RuntimeError(f"Contact '{name}' must contain finite floating-point data.")


def _histories(data, num_envs: int, *, one_partner: bool) -> tuple[torch.Tensor, torch.Tensor]:
    normal = _required(data, "normal_force_matrix_w_history")
    friction = _required(data, "friction_force_matrix_w_history")
    if normal.ndim != 5 or normal.shape[1] < 1 or normal.shape[3] < 1:
        raise RuntimeError("Contact history shape must be (N,H,1,P,3) with H >= 1 and P >= 1.")
    partners = 1 if one_partner else normal.shape[3]
    shape = (num_envs, normal.shape[1], 1, partners, 3)
    _validate(normal, shape, "normal_force_matrix_w_history")
    _validate(friction, shape, "friction_force_matrix_w_history")
    if normal.device != friction.device or normal.dtype != friction.dtype:
        raise RuntimeError("Normal/friction contact histories must share device and dtype.")
    return normal, friction


def _loaded_position(sensor, num_envs: int, dt: float, average: torch.Tensor) -> torch.Tensor:
    """Prefer PhysX's positive-normal-force weighted contact centroid, not a true COP."""
    view = getattr(sensor, "contact_view", None)
    get_contacts = getattr(view, "get_contact_data", None)
    if get_contacts is None:
        # Newton has no raw getter: its position_matrix already uses |full force|
        # weighted midpoints. Other backends retain their documented average.
        return average
    forces, points, _, _, counts, starts = get_contacts(dt=dt)
    forces = _tensor(forces, "raw contact forces").reshape(-1)
    points = _tensor(points, "raw contact points")
    counts = _tensor(counts, "raw contact counts").to(dtype=torch.long).clone()
    starts = _tensor(starts, "raw contact starts").to(dtype=torch.long).clone()
    if counts.shape != (num_envs, 1) or starts.shape != counts.shape or points.shape != (forces.numel(), 3):
        raise RuntimeError("Raw contact buffer shape mismatch; expected one sensing body and one object partner.")
    counts, starts = counts[:, 0], starts[:, 0]
    if (counts < 0).any() or ((counts > 0) & ((starts < 0) | (starts + counts > forces.numel()))).any():
        raise RuntimeError("Raw contact buffer is truncated or has invalid contact ranges.")
    # Ragged segment reduction: only gather occupied contacts, not N x capacity.
    rows = torch.repeat_interleave(torch.arange(num_envs, device=counts.device), counts)
    offsets = torch.arange(rows.numel(), device=counts.device) - torch.repeat_interleave(
        counts.cumsum(0) - counts, counts
    )
    indices = starts[rows] + offsets
    weights = forces[indices].clamp_min(0)
    positions = points[indices]
    if not torch.isfinite(weights).all() or ((weights > 0) & ~torch.isfinite(positions).all(dim=-1)).any():
        raise RuntimeError("Nonfinite loaded raw contact force or position.")
    positions = torch.where((weights > 0).unsqueeze(-1), positions, 0.0)
    numerator = torch.zeros_like(average).index_add_(0, rows, positions * weights.unsqueeze(-1))
    denominator = average.new_zeros(num_envs).index_add_(0, rows, weights)
    position = numerator / denominator.clamp_min(torch.finfo(average.dtype).tiny).unsqueeze(-1)
    position[denominator <= 0] = float("nan")
    return position


def read_contact_sample(sensor, num_envs: int, dt: float) -> ContactSample:
    """Read one object-filtered body, preserving force signs and both histories.

    ``dt`` is the physics tick (not control dt), used by the raw PhysX getter
    to convert impulses to forces. When available, raw positive-normal-force
    weighted positions exclude zero-force speculative contacts. Otherwise use
    the backend's documented average (Newton: |full force|-weighted midpoint).
    Neither is a contact manifold or exact center of pressure. A position
    without a nonzero measured normal force is invalid; invalid positions are
    NaN, never the world origin. Raw getter failures propagate, never fall back.
    """
    if num_envs < 1 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("num_envs and physics dt must be positive and dt finite.")
    data = sensor.data
    normal = _required(data, "normal_force_matrix_w")
    friction = _required(data, "friction_force_matrix_w")
    position = _required(data, "contact_pos_w")
    shape = (num_envs, 1, 1, 3)
    _validate(normal, shape, "normal_force_matrix_w")
    _validate(friction, shape, "friction_force_matrix_w")
    _validate(position, shape, "contact_pos_w", finite=False)
    history_normal, history_friction = _histories(data, num_envs, one_partner=True)
    for value in (friction, position, history_normal, history_friction):
        if value.device != normal.device or value.dtype != normal.dtype:
            raise RuntimeError("All contact sample fields must share device and dtype.")
    normal, friction, position = (value[:, 0, 0].clone() for value in (normal, friction, position))
    history_normal, history_friction = (value[:, :, 0, 0].clone() for value in (history_normal, history_friction))
    position = _loaded_position(sensor, num_envs, dt, position)
    valid = torch.isfinite(position).all(dim=-1) & (torch.linalg.vector_norm(normal, dim=-1) > 0)
    position[~valid] = float("nan")
    return ContactSample(normal, friction, normal + friction, position, valid, history_normal, history_friction)


def read_support_load(sensor, num_envs: int) -> torch.Tensor:
    """Return (N,H) sum of partner full-force norms, not norm of their sum.

    Use a sensor ON the object filtered against every disallowed support body.
    Keeping each partner separate prevents opposing support forces cancelling;
    retaining history prevents a transient support contact disappearing at 30 Hz.
    Result is newest first, matching the backend's force histories.
    """
    if num_envs < 1:
        raise ValueError("num_envs must be positive.")
    normal, friction = _histories(sensor.data, num_envs, one_partner=False)
    return torch.linalg.vector_norm(normal + friction, dim=-1).sum(dim=-1)[:, :, 0]
