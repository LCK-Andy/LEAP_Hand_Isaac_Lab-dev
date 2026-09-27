#!/usr/bin/env python3
"""Copy the local FlashSAC source and apply a guarded, reproducible replay fix.

This is a PYTHONPATH source snapshot, not an installation. Existing destinations
are only accepted when every recorded source byte and preparation setting agrees.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path.home() / "Documents/projects/FlashSAC"
DEFAULT_DESTINATION = ROOT / "logs/flashsac_validation/source/FlashSAC-v2"
UPSTREAM_PIN = "87edc9061150ae9e962dd84e6544e27a1554b3ab"
PATCH_VERSION = 2
MANIFEST = "source_manifest.json"
SELECTION = ("flash_rl", "configs", "LICENSE", "pyproject.toml", "README.md", "scripts/run_isaaclab.sh")
BASE_HASHES = {
    "flash_rl/buffers/torch_buffer.py": "663ec003992ebda3c029c963eb93db8b902bc666cdd76f451e7cb348d37439a9",
    "flash_rl/agents/flashSAC/update.py": "6134612e4ad43089dd44ce06cb6644546e864a3de13bb12525cb631e65aa30ef",
    "flash_rl/agents/utils/reward_normalization.py": "b673f5fba40468f6bb183d3cc82daa05111e95407ccb6b73de854f338c7b70b8",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args])


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Patch anchor must appear exactly once: {old!r}")
    return text.replace(old, new, 1)


def patched_files(files):
    """Require exact known source hashes before applying any patch."""
    result = dict(files)
    for name, expected in BASE_HASHES.items():
        if sha(files[name]) != expected:
            raise ValueError(f"Unsupported source hash for {name}; refusing an unreviewed patch")
    name = "flash_rl/buffers/torch_buffer.py"
    text = files[name].decode()
    patches = [
        (
            "# Mapping from numpy dtypes to torch dtypes",
            "N_STEP_DISCOUNT_VERSION = 1\n\n# Mapping from numpy dtypes to torch dtypes",
        ),
        (
            "        self._terminateds = torch.empty",
            "        self._discounts = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)\n        self._terminateds = torch.empty",
        ),
        (
            '        n_step_reward = curr_transition["reward"].clone()',
            '        n_step_reward = curr_transition["reward"].clone()\n'
            "        actual_horizon = torch.ones_like(n_step_reward, dtype=torch.int64)",
        ),
        (
            "            done_mask = done.bool()",
            "            done_mask = done.bool()\n"
            "            actual_horizon = torch.where(done_mask, 1, actual_horizon + 1)",
        ),
        (
            '        n_step_prev_transition["reward"] = n_step_reward',
            '        n_step_prev_transition["discount"] = torch.pow(\n'
            "            torch.full_like(n_step_reward, self._gamma, dtype=torch.float32), actual_horizon\n"
            "        )\n"
            '        n_step_prev_transition["reward"] = n_step_reward',
        ),
        (
            '            self._rewards[idxs] = n_step_prev_transition["reward"].to(self._rewards.dtype)',
            '            self._rewards[idxs] = n_step_prev_transition["reward"].to(self._rewards.dtype)\n'
            '            self._discounts[idxs] = n_step_prev_transition["discount"].to(self._discounts.dtype)',
        ),
        (
            '        batch["reward"] = self._rewards[idxs]',
            '        batch["reward"] = self._rewards[idxs]\n' '        batch["discount"] = self._discounts[idxs]',
        ),
        (
            '            "reward": self._rewards[:n],',
            '            "reward": self._rewards[:n],\n' '            "discount": self._discounts[:n],',
        ),
        (
            '        n = dataset["num_in_buffer"]',
            '        if "discount" not in dataset:\n'
            '            raise ValueError("Legacy replay missing discount: early timeout horizons cannot be recovered")\n'
            '        n = dataset["num_in_buffer"]',
        ),
        (
            '        self._rewards[:n] = dataset["reward"]',
            '        self._rewards[:n] = dataset["reward"]\n' '        self._discounts[:n] = dataset["discount"]',
        ),
    ]
    for old, new in patches:
        text = replace_once(text, old, new)
    # Sliced tensors retain full backing storage in torch.save unless cloned.
    for field in ("observations", "next_observations", "actions", "rewards", "discounts", "terminateds", "truncateds"):
        text = replace_once(text, f"self._{field}[:n],", f"self._{field}[:n].clone(),")
    result[name] = text.encode()
    name = "flash_rl/agents/flashSAC/update.py"
    text = files[name].decode()
    text = replace_once(
        text, "    gamma: float,\n    num_bins: int,", "    gamma: float | torch.Tensor,\n    num_bins: int,"
    )
    text = replace_once(
        text,
        "    actor_entropy = actor_entropy.reshape(-1, 1)",
        "    actor_entropy = actor_entropy.reshape(-1, 1)\n"
        "    if isinstance(gamma, torch.Tensor):\n"
        "        gamma = gamma.reshape(-1, 1)",
    )
    text = replace_once(
        text, "                gamma=gamma**n_step,", '                gamma=batch.get("discount", gamma**n_step),'
    )
    result[name] = text.encode()
    name = "flash_rl/agents/utils/reward_normalization.py"
    text = files[name].decode()
    text = replace_once(
        text,
        "    done = torch.logical_or(terminated, truncated).float()\n"
        "    new_G_r = gamma * (1.0 - done) * G_r + reward",
        "    # Include the terminal reward in this episode's statistics.\n" "    new_G_r = gamma * G_r + reward",
    )
    text = replace_once(
        text,
        "        self.G_rms.update(self.G_r)",
        "        self.G_rms.update(self.G_r)\n"
        "        # Clear carry only after recording the completed episode return.\n"
        "        self.G_r = torch.where(torch.logical_or(terminated, truncated), 0.0, self.G_r)",
    )
    result[name] = text.encode()
    return result


def collect_source(source):
    tracked = git(source, "ls-files", "-z").decode().split("\0")
    files = {}
    for name in tracked:
        if not name:
            continue
        path = source / name
        if path.is_symlink():
            raise ValueError(f"Refusing source symlink: {name}")
        if path.is_file():
            files[name] = path.read_bytes()
    selected = {
        name: data
        for name, data in files.items()
        if any(name == item or name.startswith(item + "/") for item in SELECTION)
    }
    for item in SELECTION:
        if not any(name == item or name.startswith(item + "/") for name in selected):
            raise ValueError(f"Missing required snapshot input: {item}")
    return files, selected


def prepare(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate trees")
    head = git(source, "rev-parse", "HEAD").decode().strip()
    git(source, "merge-base", "--is-ancestor", UPSTREAM_PIN, head)
    all_files, selected = collect_source(source)
    patched = patched_files(selected)
    diff = git(source, "diff", "HEAD", "--binary", "--no-ext-diff")
    manifest = {
        "schema_version": 1,
        "purpose": "Isolated PYTHONPATH source; not an installed distribution",
        "source": str(source),
        "original_base_commit": head,
        "official_upstream_pin": UPSTREAM_PIN,
        "upstream_is_ancestor": True,
        "commits_ahead_of_upstream": int(git(source, "rev-list", "--count", f"{UPSTREAM_PIN}..{head}")),
        "commits_since_upstream": git(source, "log", "--format=%H %s", f"{UPSTREAM_PIN}..{head}").decode().splitlines(),
        "remotes": git(source, "remote", "-v").decode().splitlines(),
        "tracked_dirty_status": git(source, "status", "--porcelain", "--untracked-files=no").decode().splitlines(),
        "tracked_worktree_sha256": {name: sha(data) for name, data in sorted(all_files.items())},
        "original_head_blobs": git(source, "ls-tree", "-r", head).decode().splitlines(),
        "dirty_patch_sha256": sha(diff),
        "selected_inputs_sha256": {name: sha(data) for name, data in sorted(selected.items())},
        "snapshot_files_sha256": {name: sha(data) for name, data in sorted(patched.items())},
        "patch_version": PATCH_VERSION,
        "prepare_script_sha256": sha(Path(__file__).read_bytes()),
        "patch_base_sha256": BASE_HASHES,
        "patch_files": sorted(name for name in patched if patched[name] != selected[name]),
        "legacy_replay_policy": "Reject replay checkpoints without per-transition discount",
        "reward_normalizer_boundary_fix": "Record gamma*previous_return+reward and its maximum before clearing done lanes",
    }
    patched["local_dirty.patch"] = diff
    patched[MANIFEST] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if destination.exists():
        actual = {
            str(p.relative_to(destination)): p.read_bytes()
            for p in destination.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        }
        if actual != patched:
            raise ValueError(f"Existing snapshot differs; refusing overwrite: {destination}")
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".FlashSAC-", dir=destination.parent))
    try:
        for name, data in patched.items():
            output = temporary / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            output.chmod((source / name).stat().st_mode & 0o777 if (source / name).is_file() else 0o644)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def pack_source(destination, archive):
    """Pack only manifested sources/provenance, excluding caches and run artifacts."""
    manifest = json.loads((destination / MANIFEST).read_text())
    names = sorted([*manifest["snapshot_files_sha256"], MANIFEST, "local_dirty.patch"])
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "x:gz") as tar:
        for name in names:
            tar.add(destination / name, arcname=f"FlashSAC/{name}", recursive=False)
    return sha(archive.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--tar", type=Path, help="Optionally create a source-only tar.gz")
    args = parser.parse_args()
    manifest = prepare(args.source, args.destination)
    result = {
        "snapshot": str(args.destination.resolve()),
        "manifest": str(args.destination.resolve() / MANIFEST),
        "source_head": manifest["original_base_commit"],
        "patch_version": PATCH_VERSION,
    }
    if args.tar:
        result.update(archive=str(args.tar.resolve()), archive_sha256=pack_source(args.destination, args.tar))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
