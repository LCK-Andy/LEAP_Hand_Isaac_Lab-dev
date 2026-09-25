"""Plan or serially evaluate exact-budget finals from an Allegro experiment manifest.

No simulator starts unless --run is explicit. Evaluate ALL final members (no
champion selection on test data). Validation/test/training seeds must be disjoint.
Checkpoints are trusted local torch files, never untrusted downloads.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import torch
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[1042, 1043])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[2042, 2043])
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plans = json.loads(args.manifest.read_text())
    training = {p["seed"] for p in plans}
    validation, test = set(args.validation_seeds), set(args.test_seeds)
    if training & (validation | test) or validation & test:
        parser.error("training, validation, and test seeds must be disjoint")
    if (
        min(args.episodes, args.num_envs) <= 0
        or len(validation) != len(args.validation_seeds)
        or len(test) != len(args.test_seeds)
    ):
        parser.error("positive episode/env budgets and unique seeds required")
    if len({p["steps"] for p in plans}) != 1:
        parser.error("all policies must have equal transition budgets")
    jobs = []
    seeds = args.validation_seeds if args.split == "validation" else args.test_seeds
    for plan in plans:
        checkpoint = Path(plan["final_checkpoint"]).resolve(strict=True)
        metadata = yaml.safe_load(checkpoint.with_suffix(".yaml").read_text())
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not metadata.get("terminal") or any(
            data.get("frame") != plan["steps"] or data.get("epoch") != plan["epochs"] for data in (metadata, saved)
        ):
            parser.error(f"final checkpoint does not match exact budget: {checkpoint}")
        del saved
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        for seed in seeds:
            output = args.manifest.parent / "heldout" / args.split / f"{plan['arm']}_{plan['member']}_{seed}.json"
            if output.exists():
                parser.error(f"report already exists; never overwrite heldout data: {output}")
            command = [
                args.python,
                str(Path(__file__).with_name("evaluate_allegro_pbt.py")),
                "--checkpoint",
                str(checkpoint),
                "--seed",
                str(seed),
                "--episodes",
                str(args.episodes),
                "--num_envs",
                str(args.num_envs),
                "--output",
                str(output),
                "--headless",
            ]
            jobs.append(
                {
                    "arm": plan["arm"],
                    "member": plan["member"],
                    "seed": seed,
                    "split": args.split,
                    "gpu": plan["gpu"],
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": digest,
                    "output": str(output),
                    "argv": command,
                }
            )
    print(json.dumps(jobs, indent=2), flush=True)
    if args.run:
        for job in jobs:
            subprocess.run(job["argv"], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(job["gpu"])}, check=True)
            report = json.loads(Path(job["output"]).read_text())
            if not report["completed"] or report["checkpoint_sha256"] != job["checkpoint_sha256"]:
                raise RuntimeError("Incomplete evaluation or checkpoint changed after preflight")


if __name__ == "__main__":
    main()
