"""Open Instruct code API with interchangeable Daytona and Docker execution."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rl_zero.daytona_backend import DaytonaBackend, DaytonaSettings
from rl_zero.execution_backends import DockerBackend
from rl_zero.io import object_hash
from rl_zero.rewards import decode_tests, execution_reward


def create_server(executor, trace_path, host="127.0.0.1", port=1234, workers=1):
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("Verifier workers must be a positive integer")
    identity = executor.identity
    # Retain image_id for existing clients; Daytona's value hashes the full execution identity.
    image_id = identity.get("image_id", object_hash(identity))
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
            if self.path != "/health":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            self.reply(
                200,
                {
                    "status": "healthy",
                    "backend": identity["backend"],
                    "image_id": image_id,
                    "execution_identity": identity,
                },
            )

        def do_POST(self):
            if self.path not in {"/test_program", "/test_program_stdio"}:
                self.reply(404, {"error": "Unknown test endpoint"})
                return
            start = time.time()
            record = {"time": start, "image_id": image_id, "execution_identity": identity}
            gate.acquire()  # Bound request-body memory and simultaneous remote sandboxes.
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 64 * 1024 * 1024:
                    raise ValueError("Request must be between 1 byte and 64 MiB")
                payload = json.loads(self.rfile.read(length))
                tests = decode_tests(payload["tests"])
                if isinstance(tests[0], dict) != self.path.endswith("_stdio"):
                    raise ValueError("Test format does not match the selected API endpoint")
                payload["tests"] = tests
                record.update(request_hash=object_hash(payload), program=payload["program"], tests=tests)
                result = executor.execute(payload)
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

    return ThreadingHTTPServer((host, port), Handler)


def serve(
    host="127.0.0.1",
    port=1234,
    image="rl-zero-verifier:local",
    trace_path="runs/verifier.jsonl",
    workers=None,
    backend="daytona",
    config="configs/verifiers/daytona.toml",
):
    if backend == "daytona":
        settings = DaytonaSettings.from_file(config)
        executor = DaytonaBackend(settings)
        workers = settings.workers if workers is None else workers
    elif backend == "docker":
        executor = DockerBackend(image)
        workers = 4 if workers is None else workers
    else:
        raise ValueError(f"Unknown execution backend: {backend}")
    server = create_server(executor, trace_path, host, port, workers)
    print(f"Code verifier on http://{host}:{port}; backend={backend}; workers={workers}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
