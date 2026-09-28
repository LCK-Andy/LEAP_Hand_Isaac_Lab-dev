"""Simulator-free file command protocol for the GPT-6 Astra humanoid demo.

Requests are bare command objects in ``commands/<uuid>.json``; the server writes
its response atomically to ``responses/<same uuid>.json``. No ID is added to the
command itself. Responses use ``{"ok": true, ...}`` or ``{"ok": false, "error": ...}``.
The server must validate commands again before execution and treat omitted joint
torques as zero, not as values carried over from an earlier action.

``state.json`` is a read-only snapshot with arbitrary extra fields. The action
client uses its ``joint_names`` list and optional ``max_steps`` (default 30).
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path


def atomic_write_json(path, payload):
    """Create parent directories and atomically replace a file with strict JSON.

    Serialization and filesystem failures propagate to the caller. A temporary
    file in the destination directory is removed on failure; it never matches
    the server's ``*.json`` command glob.
    """
    serialized = json.dumps(payload, allow_nan=False, ensure_ascii=True) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_command(command, joint_names, max_steps=30):
    """Return a fresh command dict with sparse torque values converted to floats.

    Args:
        command: JSON command object to validate.
        joint_names: Iterable of joint names accepted by the simulator.
        max_steps: Inclusive upper bound on action steps, default 30.

    Raises:
        ValueError: If the command or step limit is invalid. Values are never clipped.
    """
    if not isinstance(command, dict):
        raise ValueError("command must be a JSON object")
    op = command.get("op")
    if op not in ("action", "reset", "close"):
        raise ValueError("op must be action, reset, or close")
    allowed = {"op", "label", "torques", "steps"} if op == "action" else {"op", "label"}
    unknown = command.keys() - allowed
    if unknown:
        raise ValueError(f"unknown command keys: {', '.join(sorted(map(repr, unknown)))}")
    result = {"op": op}
    if "label" in command:
        label = command["label"]
        if not isinstance(label, str) or len(label) > 1000:
            raise ValueError("label must be a string of at most 1000 characters")
        result["label"] = label
    if op != "action":
        return result
    if "torques" not in command or "steps" not in command:
        raise ValueError("action requires torques and steps")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    steps = command["steps"]
    if type(steps) is not int or not 1 <= steps <= max_steps:
        raise ValueError(f"steps must be an integer in [1, {max_steps}]")
    torques = command["torques"]
    if not isinstance(torques, dict):
        raise ValueError("torques must be a JSON object mapping joint names to normalized values")
    known_joints = set(joint_names)
    normalized = {}
    for name, value in torques.items():
        if not isinstance(name, str) or name not in known_joints:
            raise ValueError(f"unknown joint: {name!r}")
        # A bounded comparison also rejects NaN/infinity and avoids float overflow
        # when a JSON integer is too large to convert to a float.
        if type(value) not in (int, float) or not -1 <= value <= 1:
            raise ValueError(f"torque for {name!r} must be a finite number in [-1, 1], not a boolean")
        normalized[name] = float(value)
    result.update(torques=normalized, steps=steps)
    return result


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _send_command(run_dir, command, timeout):
    request_id = str(uuid.uuid4())
    response_path = run_dir / "responses" / f"{request_id}.json"
    atomic_write_json(run_dir / "commands" / f"{request_id}.json", command)
    deadline = time.monotonic() + timeout
    while True:
        try:
            return _read_json(response_path)
        except FileNotFoundError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for response to request {request_id}; "
                    f"the request may still execute. Check {response_path} before retrying."
                ) from None
            time.sleep(min(0.05, remaining))


def main(argv=None):
    """Run the file-only client; return zero on success, nonzero on error.

    The global ``--timeout`` bounds response waiting (default 120 seconds).
    Timeout is not cancellation: published commands and responses are retained.
    ``status`` never sends a command or requires action-specific state fields.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory shared with the simulator process")
    parser.add_argument("--timeout", type=float, default=120.0, help="Response wait limit in seconds (default: 120)")
    subparsers = parser.add_subparsers(dest="op", required=True)
    subparsers.add_parser("status", help="Print state.json without sending a command")
    action = subparsers.add_parser("action", help="Apply sparse normalized joint torques")
    action.add_argument("--torques", required=True, help="JSON object, e.g. '{\"joint_name\":0.1}'")
    action.add_argument("--steps", type=int, required=True, help="Number of simulation steps")
    action.add_argument("--label", help="Optional audit reason (at most 1000 characters)")
    for op in ("reset", "close"):
        subparsers.add_parser(op).add_argument("--label", help="Optional audit reason (at most 1000 characters)")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be finite and greater than zero")
        if args.op == "status":
            print(json.dumps(_read_json(args.run_dir / "state.json"), allow_nan=False))
            return 0
        command = {"op": args.op}
        if args.label is not None:
            command["label"] = args.label
        joint_names, max_steps = [], 30
        if args.op == "action":
            command.update(torques=json.loads(args.torques), steps=args.steps)
            state = _read_json(args.run_dir / "state.json")
            if not isinstance(state, dict) or not isinstance(state.get("joint_names"), list):
                raise ValueError("state.json must contain a joint_names list before sending an action")
            joint_names = state["joint_names"]
            if not all(isinstance(name, str) for name in joint_names):
                raise ValueError("state.json joint_names must contain only strings")
            max_steps = state.get("max_steps", 30)
        command = validate_command(command, joint_names, max_steps)
        response = _send_command(args.run_dir, command, args.timeout)
        if not isinstance(response, dict):
            raise ValueError("response must be a JSON object")
        print(json.dumps(response, allow_nan=False))
        return 1 if response.get("ok") is False or response.get("error") else 0
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
