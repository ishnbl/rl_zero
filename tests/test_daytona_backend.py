"""Offline checks at the cloud boundary; no credentials or sandbox provisioning."""

import importlib.util
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rl_zero import sandbox_runner
from rl_zero.daytona_backend import DaytonaBackend, DaytonaSettings, build_snapshot
from rl_zero.evaluation import grade_predictions
from rl_zero.io import file_hash, write_json, write_jsonl
from rl_zero.report import compare_results
from rl_zero.verifier import create_server


class SandboxDouble:
    """In-memory transport double. It never executes the submitted program."""

    def __init__(self, error=None, output=None, cleanup_error=False):
        self.id = "sandbox-offline"
        self.files = {}
        self.deleted = False
        self.error = error
        self.output = output or {"results": [1, 0], "runtimes": [0.01, 0.01]}
        self.cleanup_error = cleanup_error
        self.fs = SimpleNamespace(upload_file=self.upload)
        self.process = SimpleNamespace(exec=self.execute)

    def upload(self, contents, path, **kwargs):
        self.files[path] = contents

    def execute(self, command, **kwargs):
        if self.error:
            raise self.error
        return SimpleNamespace(exit_code=0, result=json.dumps(self.output))

    def delete(self, **kwargs):
        self.deleted = True
        if self.cleanup_error:
            raise RuntimeError("cleanup offline")


def backend(sandbox):
    snapshot = SimpleNamespace(
        id="snapshot-pinned",
        name="test",
        image_name="image@sha256:abc",
        state="active",
        cpu=1,
        mem=1,
        disk=3,
        gpu=0,
    )
    client = SimpleNamespace(snapshot=SimpleNamespace(get=lambda _: snapshot))
    client.create = lambda params, **kwargs: setattr(client, "params", params) or sandbox
    return DaytonaBackend(DaytonaSettings(snapshot="test"), client=client), client


@unittest.skipUnless(importlib.util.find_spec("daytona"), "Install the daytona extra for SDK boundary checks")
class DaytonaContracts(unittest.TestCase):
    def test_creation_timeout_recovers_and_deletes_created_sandbox(self):
        from daytona import DaytonaTimeoutError

        sandbox = SandboxDouble()
        executor, client = backend(sandbox)

        def timeout(*args, **kwargs):
            error = DaytonaTimeoutError("startup timeout")
            error.sandbox_id = sandbox.id
            raise error

        client.create = timeout
        client.get = lambda identifier, **kwargs: sandbox
        with self.assertRaises(RuntimeError):
            executor.execute({"program": "", "tests": ["assert True"]})
        self.assertTrue(sandbox.deleted)

    def test_runner_upload_uses_the_bytes_hashed_at_startup(self):
        sandbox = SandboxDouble()
        with tempfile.TemporaryDirectory() as directory:
            runner = Path(directory) / "runner.py"
            runner.write_bytes(b"original runner bytes")
            with patch("rl_zero.daytona_backend.RUNNER", runner):
                executor, _ = backend(sandbox)
                runner.write_bytes(b"changed after startup")
                executor.execute({"program": "", "tests": ["assert True", "assert False"]})
        self.assertNotEqual(sandbox.files["/tmp/rlzero-runner.py"], b"changed after startup")

    def test_daytona_children_cannot_read_the_test_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "payload.json"
            payload.write_text(
                json.dumps(
                    {
                        "program": "not executed",
                        "tests": [{"input": "1", "output": "2"}],
                        "max_execution_time": 1,
                    }
                )
            )
            stream = io.StringIO()

            def child(*args, **kwargs):
                self.assertFalse(payload.exists())
                self.assertTrue(stream.closed)
                self.assertEqual(kwargs["user"], 65534)
                self.assertEqual(kwargs["group"], 65534)
                self.assertEqual(kwargs["extra_groups"], [])
                kwargs["stdout"].write(b"2")
                return SimpleNamespace(returncode=0)

            with (
                patch("sys.stdin", stream),
                patch("sys.stdout", io.StringIO()),
                patch("os.geteuid", return_value=0),
                patch("subprocess.run", side_effect=child),
            ):
                sandbox_runner.main(["--payload", str(payload), "--unprivileged-children"])

    def test_http_grading_keeps_daytona_identity_and_fractional_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor, _ = backend(SandboxDouble())
            server = create_server(executor, root / "trace.jsonl", port=0)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                benchmark, predictions = root / "tasks.jsonl", root / "samples.jsonl"
                write_jsonl(
                    benchmark,
                    [
                        {"id": "assertions", "domain": "code", "tests": ["assert True", "assert False"]},
                        {
                            "id": "stdio",
                            "domain": "code",
                            "tests": [{"input": "1", "output": "2"}, {"input": "2", "output": "3"}],
                        },
                    ],
                )
                write_jsonl(
                    predictions,
                    [
                        {"id": key, "completions": [{"text": "not executed offline"}]}
                        for key in ("assertions", "stdio")
                    ],
                )
                write_json(
                    str(predictions) + ".meta.json",
                    {
                        "benchmark_sha256": file_hash(benchmark),
                        "predictions_sha256": file_hash(predictions),
                        "protocol": {"samples": 1},
                        "model": {"id": "offline"},
                    },
                )
                score = grade_predictions(
                    benchmark,
                    predictions,
                    root / "score.json",
                    api_url=f"http://127.0.0.1:{server.server_port}/test_program",
                )
                self.assertEqual(score["score"], 0)  # Fractional reward is not complete task success.
                self.assertEqual(score["grader"]["kind"], "daytona-code-v1")
                self.assertEqual(score["grader"]["execution_identity"]["snapshot_id"], "snapshot-pinned")
                with self.assertRaises(ValueError):
                    compare_results(score, {**score, "grader": {"kind": "docker-code-v1"}})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_candidate_payload_is_uploaded_and_sandbox_always_deleted(self):
        sandbox = SandboxDouble()
        executor, client = backend(sandbox)
        program = 'print("literal $(do-not-run)")'
        result = executor.execute({"program": program, "tests": ["assert True", "assert False"]})
        self.assertEqual(result["results"], [1, 0])
        payloads = [json.loads(v) for k, v in sandbox.files.items() if k.endswith(".json")]
        self.assertEqual(payloads[0]["program"], program)
        self.assertTrue(sandbox.deleted)
        self.assertTrue(client.params.network_block_all)
        self.assertTrue(client.params.ephemeral)
        self.assertEqual(client.params.snapshot, "snapshot-pinned")
        self.assertFalse(client.params.public)
        self.assertGreater(client.params.ttl_minutes, 0)

    def test_execution_and_cleanup_errors_never_become_zero_rewards(self):
        for sandbox in (
            SandboxDouble(error=TimeoutError("offline timeout")),
            SandboxDouble(output={"results": [1], "runtimes": [0.1]}),
            SandboxDouble(cleanup_error=True),
        ):
            with self.subTest(sandbox=sandbox):
                executor, _ = backend(sandbox)
                with self.assertRaises((RuntimeError, TimeoutError, ValueError)):
                    executor.execute({"program": "", "tests": ["assert True", "assert True"]})
                self.assertTrue(sandbox.deleted)

    def test_snapshot_plan_does_not_create_a_cloud_client(self):
        with (
            patch("daytona.Daytona", side_effect=AssertionError("Dry runs must not create a client")),
            patch("sys.stdout", io.StringIO()),
        ):
            plan = build_snapshot(".", DaytonaSettings(snapshot="my-snapshot"), execute=False)
        self.assertEqual(plan["name"], "my-snapshot")
        self.assertIn("numpy==2.2.6", plan["packages"])
        self.assertEqual(plan["gpu"], 0)


if __name__ == "__main__":
    unittest.main()
