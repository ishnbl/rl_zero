"""Open Instruct-compatible HTTP code API backed by disposable Docker workers."""

import json
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rl_zero.io import object_hash
from rl_zero.rewards import decode_tests, execution_reward


def run_in_docker(payload, image="rl-zero-verifier:local"):
    tests = decode_tests(payload["tests"])
    timeout = float(payload.get("max_execution_time", 1))
    if not 0 < timeout <= 10 or len(tests) > 1000 or not isinstance(payload["program"], str):
        raise ValueError("Invalid program, test count (max 1000), or timeout (0 < seconds <= 10)")
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
        image,
    ]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps({**payload, "tests": tests}),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=len(tests) * timeout + 30,
            check=True,
        )
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
        raise RuntimeError("Verifier container timed out; inspect worker health") from None
    result = json.loads(completed.stdout)
    if len(result["results"]) != len(tests) or len(result["runtimes"]) != len(tests):
        raise RuntimeError("Verifier returned incomplete test results")
    execution_reward(result["results"])
    return result


def serve(
    host="127.0.0.1", port=1234, image="rl-zero-verifier:local", trace_path="runs/verifier.jsonl", workers=4
):
    # Inspect once at startup. Missing Docker/image is a setup error, never a zero reward.
    inspected = subprocess.check_output(["docker", "image", "inspect", image], text=True)
    image_id = json.loads(inspected)[0]["Id"]
    gate, log_lock = threading.BoundedSemaphore(workers), threading.Lock()
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(
                200 if self.path == "/health" else 404,
                {"status": "healthy", "backend": "docker", "image_id": image_id},
            )

        def do_POST(self):
            if self.path not in {"/test_program", "/test_program_stdio"}:
                self.reply(404, {"error": "Unknown test endpoint"})
                return
            start = time.time()
            record = {"time": start, "image_id": image_id}
            gate.acquire()  # Bound both request-body memory and running containers.
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 64 * 1024 * 1024:
                    raise ValueError("Request must be between 1 byte and 64 MiB")
                payload = json.loads(self.rfile.read(length))
                tests = decode_tests(payload["tests"])
                if isinstance(tests[0], dict) != self.path.endswith("_stdio"):
                    raise ValueError("Test format does not match the selected API endpoint")
                payload["tests"] = tests
                record.update(
                    request_hash=object_hash(payload), program=payload["program"], tests=payload["tests"]
                )
                result = run_in_docker(payload, image=image_id)
                record.update(result=result, reward=execution_reward(result["results"]))
                self.reply(200, result)
            except (ValueError, KeyError) as error:
                record["error"] = str(error)
                self.reply(400, {"error": str(error)})
            except Exception as error:
                record["infrastructure_error"] = str(error)
                self.reply(503, {"error": str(error)})
            finally:
                gate.release()
                record["wall_seconds"] = time.time() - start
                with log_lock, trace_path.open("a") as output:
                    output.write(json.dumps(record) + "\n")

    print(f"Code verifier on http://{host}:{port}; Docker image {image_id}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
