"""Standard-library-only tests for the humanoid demo's file command interface."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "scripts/demos/astra_humanoid_protocol.py"
spec = importlib.util.spec_from_file_location("astra_humanoid_protocol", PROTOCOL_PATH)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


class ValidateCommandTests(unittest.TestCase):
    def test_action_returns_independent_sparse_normalized_command(self):
        command = {"op": "action", "torques": {"hip": -1, "knee": 1}, "steps": 30, "label": "boundary probe"}
        validated = protocol.validate_command(command, ["hip", "knee", "ankle"])
        self.assertEqual(
            validated,
            {"op": "action", "torques": {"hip": -1.0, "knee": 1.0}, "steps": 30, "label": "boundary probe"},
        )
        self.assertIsInstance(validated["torques"]["hip"], float)
        self.assertIsNot(validated, command)
        self.assertIsNot(validated["torques"], command["torques"])
        self.assertIsInstance(command["torques"]["hip"], int)

    def test_empty_torques_and_step_bounds_are_valid(self):
        for steps in (1, 30):
            with self.subTest(steps=steps):
                command = {"op": "action", "torques": {}, "steps": steps}
                self.assertEqual(protocol.validate_command(command, []), command)
        command = {"op": "action", "torques": {"hip": 0}, "steps": 2}
        self.assertEqual(protocol.validate_command(command, ["hip"], max_steps=2), command)
        with self.assertRaises(ValueError):
            protocol.validate_command(command, ["hip"], max_steps=1)

    def test_invalid_command_shapes_operations_and_keys_are_rejected(self):
        for command in (
            None,
            [],
            "action",
            1,
            {},
            {"op": None},
            {"op": []},
            {"op": True},
            {"op": "status"},
            {"op": "ACTION"},
            {"op": "reset", "steps": 1},
            {"op": "close", "torques": {}},
            {"op": "reset", "id": "uuid"},
            {"op": "action", "torques": {}, "steps": 1, "surprise": 0},
            {"op": "action", "torques": {}, "steps": 1, 0: "bad key"},
            {"op": "action"},
            {"op": "action", "torques": {}},
            {"op": "action", "steps": 1},
        ):
            with self.subTest(command=command), self.assertRaises(ValueError):
                protocol.validate_command(command, ["hip"])

    def test_torques_must_be_a_dict_of_known_joints(self):
        for torques in (None, [], 1, "{}", {"unknown": 0.1}, {1: 0.1}):
            with self.subTest(torques=torques), self.assertRaises(ValueError):
                protocol.validate_command({"op": "action", "torques": torques, "steps": 1}, ["hip"])

    def test_torques_reject_nonfinite_boolean_nonnumeric_and_out_of_range_values(self):
        for value in (
            True,
            False,
            None,
            "0.1",
            [],
            {},
            float("nan"),
            float("inf"),
            -float("inf"),
            -1.000001,
            1.000001,
            10**1000,
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.validate_command({"op": "action", "torques": {"hip": value}, "steps": 1}, ["hip"])

    def test_steps_and_limit_require_positive_nonboolean_integers(self):
        for steps in (True, False, None, "1", 1.0, 0, -1, 31, [], float("inf")):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                protocol.validate_command({"op": "action", "torques": {}, "steps": steps}, [])
        for limit in (True, False, None, "30", 30.0, 0, -1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                protocol.validate_command({"op": "action", "torques": {}, "steps": 1}, [], max_steps=limit)

    def test_reset_close_and_labels(self):
        for op in ("action", "reset", "close"):
            base = {"op": op}
            if op == "action":
                base.update(torques={}, steps=1)
            self.assertEqual(protocol.validate_command(base, []), base)
            for label in ("", "reason\nwith Unicode \u03b8", "x" * 1000):
                command = {**base, "label": label}
                with self.subTest(op=op, label_length=len(label)):
                    self.assertEqual(protocol.validate_command(command, []), command)
            for label in (None, True, 1, [], {}, "x" * 1001):
                with self.subTest(op=op, label=label), self.assertRaises(ValueError):
                    protocol.validate_command({**base, "label": label}, [])


class AtomicWriteTests(unittest.TestCase):
    def test_creates_parent_directories_and_replaces_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "state.json"
            protocol.atomic_write_json(str(path), {"old": True})
            payload = {"joint_names": ["hip"], "extra": {"arbitrary": [1, None, "\u03b8"]}}
            real_replace = os.replace

            def inspect_replace(source, destination):
                source, destination = Path(source), Path(destination)
                self.assertEqual(destination, path)
                self.assertEqual(source.parent, path.parent)
                self.assertNotEqual(source.suffix, ".json")
                self.assertEqual(json.loads(path.read_text()), {"old": True})
                self.assertEqual(json.loads(source.read_text()), payload)
                real_replace(source, destination)

            with mock.patch.object(protocol.os, "replace", side_effect=inspect_replace):
                self.assertIsNone(protocol.atomic_write_json(path, payload))
            self.assertEqual(json.loads(path.read_text()), payload)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_invalid_serialization_preserves_old_file_without_temporary_debris(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            protocol.atomic_write_json(path, {"old": True})
            original = path.read_bytes()
            for bad in (float("nan"), float("inf"), -float("inf"), object()):
                with self.subTest(value=bad), self.assertRaises((ValueError, TypeError)):
                    protocol.atomic_write_json(path, {"nested": [bad]})
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(list(path.parent.iterdir()), [path])

    def test_replace_failure_preserves_old_file_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            protocol.atomic_write_json(path, {"old": True})
            with mock.patch.object(protocol.os, "replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    protocol.atomic_write_json(path, {"new": True})
            self.assertEqual(json.loads(path.read_text()), {"old": True})
            self.assertEqual(list(path.parent.iterdir()), [path])


class ClientTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name) / "run"
        self.state = {"joint_names": ["hip", "knee"], "startup": {"renderer": "off"}, "extra": [None, 1]}
        protocol.atomic_write_json(self.run_dir / "state.json", self.state)

    def argv(self, *arguments, timeout="2"):
        return [
            sys.executable,
            "-B",
            "-S",
            str(PROTOCOL_PATH),
            "--run-dir",
            str(self.run_dir),
            "--timeout",
            timeout,
            *arguments,
        ]

    def client(self, *arguments, timeout="2"):
        return subprocess.run(self.argv(*arguments, timeout=timeout), capture_output=True, text=True, timeout=5)

    def start_client(self, *arguments):
        process = subprocess.Popen(self.argv(*arguments), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def cleanup():
            if process.poll() is None:
                process.kill()
            process.communicate()

        self.addCleanup(cleanup)
        return process

    def wait_for_requests(self, count):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            paths = list((self.run_dir / "commands").glob("*.json"))
            if len(paths) >= count:
                return paths
            time.sleep(0.01)
        self.fail(f"Expected {count} command files")

    def test_status_only_reads_state_and_preserves_arbitrary_fields(self):
        before = (self.run_dir / "state.json").read_bytes()
        result = self.client("status")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), self.state)
        self.assertEqual((self.run_dir / "state.json").read_bytes(), before)
        self.assertEqual(list(self.run_dir.iterdir()), [self.run_dir / "state.json"])

    def test_action_round_trip_uses_matching_uuid_and_full_command(self):
        unrelated = self.run_dir / "responses" / f"{uuid.uuid4()}.json"
        protocol.atomic_write_json(unrelated, {"ok": True, "wrong_request": True})
        process = self.start_client("action", "--torques", '{"hip":0.1}', "--steps", "3", "--label", "balance probe")
        request = self.wait_for_requests(1)[0]
        self.assertEqual(str(uuid.UUID(request.stem)), request.stem)
        self.assertEqual(uuid.UUID(request.stem).version, 4)
        self.assertEqual(
            json.loads(request.read_text()),
            {"op": "action", "torques": {"hip": 0.1}, "steps": 3, "label": "balance probe"},
        )
        self.assertIsNone(process.poll(), "An unrelated response must not complete this request")
        response = {"ok": True, "state": {"steps": 3, "read_only_extra": True}}
        protocol.atomic_write_json(self.run_dir / "responses" / request.name, response)
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(json.loads(stdout), response)
        self.assertTrue(unrelated.exists())

    def test_concurrent_reset_close_clients_keep_separate_requests_and_responses(self):
        (self.run_dir / "state.json").unlink()  # Recovery/close must also work during startup.
        reset = self.start_client("reset", "--label", "recover")
        close = self.start_client("close")
        requests = self.wait_for_requests(2)
        self.assertNotEqual(requests[0].stem, requests[1].stem)
        for request in requests:
            command = json.loads(request.read_text())
            expected = {"op": "reset", "label": "recover"} if command["op"] == "reset" else {"op": "close"}
            self.assertEqual(command, expected)
            protocol.atomic_write_json(self.run_dir / "responses" / request.name, {"ok": True, "op": command["op"]})
        for process, op in ((reset, "reset"), (close, "close")):
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertEqual(json.loads(stdout), {"ok": True, "op": op})

    def test_timeout_is_bounded_warns_of_late_execution_and_keeps_request(self):
        started = time.monotonic()
        result = self.client("reset", timeout="0.05")
        self.assertLess(time.monotonic() - started, 3)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("request may still execute", result.stderr)
        requests = list((self.run_dir / "commands").glob("*.json"))
        self.assertEqual(len(requests), 1)
        self.assertIn(requests[0].stem, result.stderr)
        self.assertEqual(json.loads(requests[0].read_text()), {"op": "reset"})

    def test_error_response_is_printed_and_exits_nonzero(self):
        process = self.start_client("reset")
        request = self.wait_for_requests(1)[0]
        response = {"ok": False, "error": "server refused reset", "extra": {"step": 5}}
        protocol.atomic_write_json(self.run_dir / "responses" / request.name, response)
        stdout, stderr = process.communicate(timeout=5)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(json.loads(stdout), response)
        self.assertEqual(stderr, "")

    def test_malformed_response_is_reported_without_traceback(self):
        process = self.start_client("reset")
        request = self.wait_for_requests(1)[0]
        response_path = self.run_dir / "responses" / request.name
        response_path.parent.mkdir()
        temporary = response_path.with_suffix(".tmp")
        temporary.write_text("not JSON", encoding="utf-8")
        os.replace(temporary, response_path)
        stdout, stderr = process.communicate(timeout=5)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(stdout, "")
        self.assertIn("Error:", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_invalid_cli_actions_do_not_publish_requests(self):
        for arguments in (
            ["action", "--steps", "1"],
            ["action", "--torques", "{}"],
            ["action", "--torques", "bad JSON", "--steps", "1"],
            ["action", "--torques", "[]", "--steps", "1"],
            ["action", "--torques", '{"unknown":0.1}', "--steps", "1"],
            ["action", "--torques", '{"hip":NaN}', "--steps", "1"],
            ["action", "--torques", '{"hip":true}', "--steps", "1"],
            ["action", "--torques", '{"hip":1.01}', "--steps", "1"],
            ["action", "--torques", "{}", "--steps", "31"],
            ["action", "--torques", "{}", "--steps", "1.0"],
            ["action", "--torques", "{}", "--steps", "1", "--label", "x" * 1001],
            ["reset", "--steps", "1"],
        ):
            with self.subTest(arguments=arguments):
                result = self.client(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse((self.run_dir / "commands").exists())

    def test_action_honors_advertised_step_limit(self):
        protocol.atomic_write_json(self.run_dir / "state.json", {**self.state, "max_steps": 2})
        result = self.client("action", "--torques", "{}", "--steps", "3")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("[1, 2]", result.stderr)
        self.assertFalse((self.run_dir / "commands").exists())

    def test_action_rejects_missing_or_malformed_joint_metadata_without_publishing(self):
        for state in ({"startup": True}, [], {"joint_names": "hip"}, {"joint_names": [None]}):
            with self.subTest(state=state):
                protocol.atomic_write_json(self.run_dir / "state.json", state)
                result = self.client("action", "--torques", "{}", "--steps", "1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("joint_names", result.stderr)
                self.assertFalse((self.run_dir / "commands").exists())

    def test_status_does_not_require_joint_metadata_during_startup(self):
        state = {"status": "starting", "diagnostics": {"message": "assets loading"}}
        protocol.atomic_write_json(self.run_dir / "state.json", state)
        result = self.client("status")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), state)
        self.assertFalse((self.run_dir / "commands").exists())

    def test_missing_or_malformed_state_is_an_error_without_creating_queue(self):
        path = self.run_dir / "state.json"
        path.unlink()
        for op in (("status",), ("action", "--torques", "{}", "--steps", "1")):
            result = self.client(*op)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Traceback", result.stderr)
        path.write_text("{broken", encoding="utf-8")
        result = self.client("status")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.run_dir / "commands").exists())

    def test_invalid_timeout_cannot_publish_an_unbounded_request(self):
        for timeout in ("nan", "inf", "-inf", "0", "-1"):
            with self.subTest(timeout=timeout):
                result = self.client("reset", timeout=timeout)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.run_dir / "commands").exists())

    def test_import_is_silent_and_does_not_parse_cli_or_create_files(self):
        code = (
            "import importlib.util, sys; "
            f"spec = importlib.util.spec_from_file_location('protocol', {str(PROTOCOL_PATH)!r}); "
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "assert not any(name.split('.')[0] in ('isaaclab', 'isaacsim', 'torch', 'numpy', 'pxr') "
            "for name in sys.modules)"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-B", "-S", "-c", code, "--not-a-real-option"],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, "")
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
