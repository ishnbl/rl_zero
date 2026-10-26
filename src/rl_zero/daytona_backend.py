"""Remote Docker-image execution through Daytona; no local Docker daemon needed."""

import hashlib
import importlib.metadata
import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path

from rl_zero.config import load_toml
from rl_zero.execution_backends import validate_payload, validate_result

RUNNER = Path(__file__).with_name("sandbox_runner.py")


@dataclass(frozen=True)
class DaytonaSettings:
    snapshot: str = "rl-zero-verifier-v1"
    expected_snapshot_id: str = ""
    create_timeout: int = 120
    transfer_timeout: int = 120
    delete_timeout: int = 60
    ttl_minutes: int = 30
    workers: int = 1
    base_image: str = "python:3.12-slim"
    cpu: int = 1
    memory: int = 1
    disk: int = 3

    def __post_init__(self):
        if not self.snapshot.strip():
            raise ValueError("Configure a Daytona snapshot name or ID")
        for key in (
            "create_timeout",
            "transfer_timeout",
            "delete_timeout",
            "ttl_minutes",
            "workers",
            "cpu",
            "memory",
            "disk",
        ):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Daytona {key} must be a positive integer")

    @classmethod
    def from_file(cls, path):
        try:
            return cls(**load_toml(path))
        except TypeError as error:
            raise ValueError(f"Invalid Daytona configuration: {error}") from error


def sdk():
    try:
        import daytona
    except ImportError as error:
        raise ValueError("Install Daytona support: python -m pip install -e '.[daytona]'") from error
    return daytona


class DaytonaBackend:
    def __init__(self, settings, client=None):
        self.settings = settings
        self.runner_bytes = RUNNER.read_bytes()
        self.client = client if client is not None else sdk().Daytona()
        # Read metadata only. Starting the verifier does not provision a sandbox.
        snapshot = self.client.snapshot.get(settings.snapshot)
        if settings.expected_snapshot_id and snapshot.id != settings.expected_snapshot_id:
            raise ValueError("Daytona snapshot ID changed; update the explicit environment pin")
        if str(snapshot.state).lower() not in {"active", "snapshotstate.active"}:
            raise ValueError(f"Daytona snapshot is not active: {snapshot.state}")
        if snapshot.gpu:
            raise ValueError("Code verification requires a CPU-only Daytona snapshot")
        self.identity = {
            "backend": "daytona",
            "snapshot_id": snapshot.id,
            "image_name": snapshot.image_name,
            "runner_sha256": hashlib.sha256(self.runner_bytes).hexdigest(),
            "candidate_uid": 65534,
            "sdk_version": importlib.metadata.version("daytona"),
            "cpu": snapshot.cpu,
            "memory_gib": snapshot.mem,
            "disk_gib": snapshot.disk,
            "network_block_all": True,
            "ttl_minutes": settings.ttl_minutes,
        }

    def execute(self, payload):
        payload = validate_payload(payload)
        settings = self.settings
        execution_timeout = math.ceil(len(payload["tests"]) * payload["max_execution_time"] + 30)
        if (
            execution_timeout + settings.create_timeout + 2 * settings.transfer_timeout
            >= settings.ttl_minutes * 60
        ):
            raise ValueError("Test budget exceeds sandbox TTL; increase ttl_minutes explicitly")
        params = sdk().CreateSandboxFromSnapshotParams(
            snapshot=self.identity["snapshot_id"],
            name="rlzero-" + uuid.uuid4().hex,
            language="python",
            os_user="root",
            public=False,
            network_block_all=True,
            ephemeral=True,
            auto_stop_interval=0,
            auto_delete_interval=0,
            ttl_minutes=settings.ttl_minutes,
            labels={"project": "rl-zero-lab", "purpose": "code-verifier"},
        )
        try:
            sandbox = self.client.create(params, timeout=settings.create_timeout)
        except Exception as error:
            identifier = getattr(error, "sandbox_id", None)
            if identifier:
                try:
                    self.client.get(identifier, request_timeout=settings.delete_timeout).delete(
                        timeout=settings.delete_timeout, wait=True
                    )
                except Exception as cleanup_error:
                    raise RuntimeError(
                        f"Creation failed and cleanup failed for Daytona sandbox {identifier}: "
                        f"{cleanup_error}; creation_error={error!r}"
                    ) from cleanup_error
            raise RuntimeError(
                f"Daytona sandbox creation failed ({identifier or params.name}): {error}"
            ) from error
        execution_error = None
        try:
            sandbox.fs.upload_file(
                self.runner_bytes, "/tmp/rlzero-runner.py", timeout=settings.transfer_timeout
            )
            sandbox.fs.upload_file(
                json.dumps(payload).encode(), "/tmp/rlzero-payload.json", timeout=settings.transfer_timeout
            )
            # Candidate text is uploaded as data, never interpolated into shell commands.
            response = sandbox.process.exec(
                "python3 -I /tmp/rlzero-runner.py --payload /tmp/rlzero-payload.json --unprivileged-children",
                timeout=execution_timeout,
            )
            if response.exit_code != 0:
                raise RuntimeError(
                    f"Daytona verifier runner exited {response.exit_code} in sandbox {sandbox.id}"
                )
            try:
                result = validate_result(json.loads(response.result), len(payload["tests"]))
            except (ValueError, TypeError) as error:
                raise RuntimeError("Daytona runner returned malformed JSON") from error
            return {**result, "sandbox_id": sandbox.id, "snapshot_id": self.identity["snapshot_id"]}
        except BaseException as error:
            execution_error = error
            raise
        finally:
            try:
                sandbox.delete(timeout=settings.delete_timeout, wait=True)
            except Exception as error:
                # Preserve the sandbox ID for manual cleanup, even if execution also failed.
                raise RuntimeError(
                    f"Failed to delete Daytona sandbox {sandbox.id}: {error}; "
                    f"execution_error={execution_error!r}"
                ) from error


def snapshot_plan(root, settings):
    packages = [
        line.strip()
        for line in (Path(root) / "docker/requirements-verifier.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return {
        "name": settings.snapshot,
        "base_image": settings.base_image,
        "packages": packages,
        "cpu": settings.cpu,
        "memory_gib": settings.memory,
        "disk_gib": settings.disk,
        "gpu": 0,
        "entrypoint": ["sleep", "infinity"],
        "build_location": "Daytona (remote)",
    }


def build_snapshot(root, settings, execute=False):
    plan = snapshot_plan(root, settings)
    print(json.dumps(plan, indent=2))
    if not execute:
        return plan
    module = sdk()
    client = module.Daytona()
    image = module.Image.base(plan["base_image"]).pip_install(plan["packages"])
    snapshot = client.snapshot.create(
        module.CreateSnapshotParams(
            name=plan["name"],
            image=image,
            entrypoint=plan["entrypoint"],
            resources=module.Resources(cpu=settings.cpu, memory=settings.memory, disk=settings.disk, gpu=0),
        ),
        timeout=600,
    )
    result = {"snapshot_id": snapshot.id, "name": snapshot.name, "image_name": snapshot.image_name}
    print(json.dumps(result, indent=2))
    return result
