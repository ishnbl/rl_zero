"""Training is a saved, inspectable plan before it is a subprocess."""

import datetime
import importlib.metadata
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

from rl_zero.data import verify_prepared
from rl_zero.io import file_hash, write_json
from rl_zero.upstream import git, validate_flags, verify_checkout


def flags_to_argv(flags):
    result = []
    for name, value in flags.items():
        if value is None:
            continue
        result.append(f"--{name}")
        values = value if isinstance(value, list) else [value]
        result.extend(str(v) for v in values)
    return result


def build_plan(root, config, run_dir):
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    method = config["experiment"]["method"]
    flags = dict(config["trainer"])
    data = (root / config["experiment"]["data_dir"]).resolve()
    flags.update(
        dataset_mixer_list=[str(data / f"train.{method}.jsonl"), "1.0"],
        dataset_mixer_list_splits=["train"],
        output_dir=str(run_dir / "checkpoints"),
        exp_name=config["experiment"]["name"],
    )
    if method == "rlvr":
        flags.update(
            dataset_mixer_eval_list=[str(data / "validation.rlvr.jsonl"), "1.0"],
            dataset_mixer_eval_list_splits=["train"],
            rollouts_save_path=str(run_dir / "rollouts"),
        )
        flags.setdefault("checkpoint_state_dir", str(run_dir / "training_state"))
    # Publication defaults in upstream are True. This adapter always runs locally.
    for key in (
        "push_to_hub",
        "try_auto_save_to_beaker",
        "try_launch_beaker_eval_jobs_on_weka" if method == "rlvr" else "try_launch_beaker_eval_jobs",
    ):
        if flags.get(key) is not False:
            raise ValueError(f"Local experiment recipes must explicitly set {key}=false")
    checkout = root / "vendor/open-instruct"
    runtime = config["runtime"]
    command = [
        runtime["uv"],
        "run",
        "--project",
        str(checkout),
        "--frozen",
        "--no-default-groups",
        "--group",
        "dev",
        "--group",
        runtime["cuda_group"],
    ]
    if method == "sft":
        command += [
            "accelerate",
            "launch",
            "--config_file",
            str(root / runtime["accelerate_config"]),
            "--num_processes",
            str(runtime["sft_processes"]),
        ]
    else:
        command += ["python"]
    command += [
        str(root / "src/rl_zero/train_entry.py"),
        "--trainer",
        str(checkout / f"open_instruct/{'grpo_fast' if method == 'rlvr' else 'finetune'}.py"),
        *flags_to_argv(flags),
    ]
    environment = {
        **config["environment"],
        "RL_ZERO_RUN_DIR": str(run_dir),
        "WANDB_DIR": str(run_dir),
        "PYTHONPATH": str(root / "src") + os.pathsep + str(checkout),
    }
    return {
        "root": str(root),
        "run_dir": str(run_dir),
        "method": method,
        "config": config,
        "data_dir": str(data),
        "trainer": flags,
        "environment": environment,
        "command": command,
        "shell_command": shlex.join(command),
        "prepared_manifest_sha256": file_hash(data / "manifest.json")
        if (data / "manifest.json").exists()
        else None,
    }


def save_plan(plan):
    run = Path(plan["run_dir"])
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "plan.json", plan)
    (run / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + f"cd {shlex.quote(plan['root'])}\n"
        + "# Environment overrides are recorded in plan.json.\n"
        + "export "
        + " ".join(f"{k}={shlex.quote(v)}" for k, v in plan["environment"].items())
        + "\n"
        + plan["shell_command"]
        + "\n"
    )
    write_json(run / "status.json", {"state": "planned"})


def execute_plan(path):
    plan = json.loads(Path(path).read_text())
    root, run = Path(plan["root"]), Path(plan["run_dir"])
    status = json.loads((run / "status.json").read_text())
    if status["state"] != "planned":
        raise ValueError("This run has already started. Create a new run plan for a resume.")
    provenance = verify_checkout(root)
    if provenance["commit"] != plan["config"]["sources"]["open_instruct"]["commit"]:
        raise ValueError("Trainer source pin changed after planning; create a new plan")
    validate_flags(provenance["path"], plan["trainer"], method=plan["method"])
    manifest = verify_prepared(plan["data_dir"])
    if plan.get("prepared_manifest_sha256") is None:
        raise ValueError("Data was absent when this plan was created. Prepare it and create a fresh plan.")
    if file_hash(Path(plan["data_dir"]) / "manifest.json") != plan["prepared_manifest_sha256"]:
        raise ValueError("Prepared data manifest changed after planning; create a fresh plan")
    if plan["method"] == "sft" and manifest["cohort"] != "matched":
        raise ValueError("SFT requires a matched cohort containing reference solutions")
    if not shutil.which(plan["command"][0]):
        raise ValueError("uv is missing; install uv before executing a training plan")
    if plan["method"] == "rlvr":
        health = plan["trainer"]["code_api_url"].rsplit("/", 1)[0] + "/health"
        with urllib.request.urlopen(health, timeout=10) as response:
            if json.load(response).get("status") != "healthy":
                raise ValueError("Code verifier is not healthy")
    write_json(run / "data_manifest.json", manifest)
    source_files = list((root / "src/rl_zero").glob("*.py")) + list((root / "configs").rglob("*"))
    hashes = {str(p.relative_to(root)): file_hash(p) for p in source_files if p.is_file()}
    for source in source_files:
        if source.is_file():
            target = run / "source" / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    write_json(
        run / "provenance.json",
        {
            "trainer": provenance,
            "source_hashes": hashes,
            "python": sys.version,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "repository_status": git(root, "status", "--porcelain"),
        },
    )
    environment = {**os.environ, **plan["environment"]}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(run / "status.json", {"state": "running", "started": now})
    try:
        with (run / "train.log").open("w") as log:
            process = subprocess.Popen(
                plan["command"],
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            stop_monitor = threading.Event()

            def monitor_verifier():
                while not stop_monitor.wait(1):
                    if process.poll() is not None:
                        return
                    if any(run.glob("verifier_failure_*.json")):
                        os.killpg(process.pid, signal.SIGTERM)
                        return

            monitor = threading.Thread(target=monitor_verifier, daemon=True)
            monitor.start()
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            finally:
                stop_monitor.set()
                monitor.join(timeout=2)
        if any(run.glob("verifier_failure_*.json")):
            raise RuntimeError("Code verifier failed; run invalidated. See verifier_failure_*.json")
        state = "completed" if code == 0 else "failed"
        write_json(run / "status.json", {"state": state, "exit_code": code, "started": now})
        if code:
            raise RuntimeError(f"Trainer exited {code}; see {run / 'train.log'}")
    except BaseException as error:
        write_json(run / "status.json", {"state": "failed", "error": str(error), "started": now})
        raise
