"""Small execution boundary shared by the HTTP service and sandbox providers."""

import json
import math
import subprocess
import uuid

from rl_zero.rewards import decode_tests, execution_reward


def validate_payload(payload):
    tests = decode_tests(payload["tests"])
    timeout = float(payload.get("max_execution_time", 1))
    if not 0 < timeout <= 10 or len(tests) > 1000 or not isinstance(payload["program"], str):
        raise ValueError("Invalid program, test count (max 1000), or timeout (0 < seconds <= 10)")
    return {"program": payload["program"], "tests": tests, "max_execution_time": timeout}


def validate_result(result, count):
    try:
        if len(result["results"]) != count or len(result["runtimes"]) != count:
            raise ValueError("Incomplete test vector")
        execution_reward(result["results"])
        if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in result["runtimes"]):
            raise ValueError("Invalid test runtimes")
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Verifier returned invalid results: {error}") from error
    return result


class DockerBackend:
    def __init__(self, image="rl-zero-verifier:local"):
        inspected = subprocess.check_output(["docker", "image", "inspect", image], text=True)
        self.identity = {"backend": "docker", "image_id": json.loads(inspected)[0]["Id"]}

    def execute(self, payload):
        payload = validate_payload(payload)
        name = "rlzero-" + uuid.uuid4().hex
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--memory",
            "512m",
            "--memory-swap",
            "512m",
            "--cpus",
            "1",
            "--pids-limit",
            "64",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "-i",
            self.identity["image_id"],
        ]
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(payload),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=len(payload["tests"]) * payload["max_execution_time"] + 30,
                check=True,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
            raise RuntimeError("Verifier container timed out; inspect worker health") from None
        return validate_result(json.loads(completed.stdout), len(payload["tests"]))
