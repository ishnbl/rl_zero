"""Small explicit commands; --execute is required for training and inference."""

import argparse
import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from rl_zero.benchmarks import prepare_benchmark, prepare_hub_code
from rl_zero.config import resolve_config
from rl_zero.data import prepare_code_file
from rl_zero.demo import run_demo
from rl_zero.diagnostics import dashboard
from rl_zero.evaluation import grade_predictions
from rl_zero.launch import build_plan, execute_plan, save_plan
from rl_zero.length_control import filter_lengths
from rl_zero.report import write_comparison
from rl_zero.upstream import bootstrap, template_text, validate_flags, verify_checkout
from rl_zero.verifier import serve


def root_directory():
    root = Path.cwd()
    for candidate in (root, *root.parents):
        if (candidate / "configs/sources.lock.json").exists():
            return candidate
    raise ValueError("Run rl-zero from this repository, or use --root /path/to/rl_zero")


def upstream_command(root, module, arguments, execute=False, cuda_group="cuda12"):
    verify_checkout(root)
    command = [
        "uv",
        "run",
        "--project",
        str(root / "vendor/open-instruct"),
        "--frozen",
        "--no-default-groups",
        "--group",
        "dev",
        "--group",
        cuda_group,
        "python",
        "-m",
        module,
        *arguments,
    ]
    print(shlex.join(command))
    if execute:
        environment = {
            **os.environ,
            "PYTHONPATH": str(root / "src") + os.pathsep + str(root / "vendor/open-instruct"),
        }
        subprocess.run(command, env=environment, cwd=root, check=True)


def parser():
    result = argparse.ArgumentParser(prog="rl-zero", description=__doc__)
    result.add_argument("--root", type=Path)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("bootstrap", help="Fetch the pinned trainer source, without installing GPU packages")
    commands.add_parser("doctor", help="CPU-only dependency and source checks")
    validate = commands.add_parser("validate", help="Static checks for all experiment recipes")
    validate.add_argument("--hardware")
    data = commands.add_parser("prepare-code", help="Download pinned Dolci data or normalize local JSONL")
    data.add_argument("--input", type=Path)
    data.add_argument("--output", type=Path, default=Path("data/code"))
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--validation-fraction", type=float, default=0.05)
    data.add_argument("--test-fraction", type=float, default=0.05)
    data.add_argument(
        "--cohort",
        choices=["matched", "full_rlvr"],
        default="matched",
        help="matched keeps solution-backed tasks for RLVR/SFT; full_rlvr includes tasks without solutions",
    )
    data.add_argument("--unsupported-tests", choices=["error", "exclude"], default="error")
    data.add_argument("--conflicting-prompts", choices=["error", "exclude"], default="error")
    bench = commands.add_parser("prepare-benchmark")
    bench.add_argument("name", choices=["math500", "gsm8k", "mbpp", "humaneval"])
    bench.add_argument("--output", type=Path)
    length = commands.add_parser("filter-lengths", help="Tokenizer-only common RLVR/SFT training cohort")
    length.add_argument("--input", type=Path, default=Path("data/code"))
    length.add_argument("--output", type=Path, default=Path("data/code_controlled"))
    length.add_argument("--max-prompt-tokens", type=int, default=2048)
    length.add_argument("--max-total-tokens", type=int, default=18432)
    plan = commands.add_parser(
        "plan", help="Save a training command and resolved configuration; never trains"
    )
    plan.add_argument("experiment", type=Path)
    plan.add_argument("--hardware")
    plan.add_argument("--seed", type=int)
    plan.add_argument("--run-dir", type=Path)
    launch = commands.add_parser("launch", help="Inspect a saved plan, or explicitly execute it")
    launch.add_argument("plan", type=Path)
    launch.add_argument("--execute", action="store_true")
    generation = commands.add_parser("generate", help="Plan/execute inference on a benchmark snapshot")
    generation.add_argument("--model", required=True, help="base, math, HF ID, or exported checkpoint path")
    generation.add_argument("--revision")
    generation.add_argument("--benchmark", type=Path, required=True)
    generation.add_argument("--output", type=Path, required=True)
    generation.add_argument("--config", type=Path, default=Path("configs/evaluation.toml"))
    generation.add_argument("--cuda-group", default="cuda12", choices=["cuda12", "cuda13"])
    generation.add_argument("--execute", action="store_true")
    grade = commands.add_parser("grade", help="Grade saved samples; no model is loaded")
    for name in ("benchmark", "predictions", "output"):
        grade.add_argument("--" + name, type=Path, required=True)
    grade.add_argument("--k", type=int, default=1)
    grade.add_argument("--api-url", default="http://127.0.0.1:1234/test_program")
    grade.add_argument("--upstream", action="store_true", help="Use pinned environment for math grading")
    compare = commands.add_parser("compare", help="Paired capability/forgetting comparison")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("dashboard", help="Write an interactive HTML metric viewer")
    inspect.add_argument("run_dir", type=Path)
    inspect.add_argument("--output", type=Path)
    verifier = commands.add_parser("verifier", help="Serve the Docker-backed code reward API")
    verifier.add_argument("--host", default="127.0.0.1")
    verifier.add_argument("--port", type=int, default=1234)
    verifier.add_argument("--image", default="rl-zero-verifier:local")
    verifier.add_argument("--trace-path", default="runs/verifier.jsonl")
    verifier.add_argument("--workers", type=int, default=4)
    checkpoints = commands.add_parser(
        "checkpoints", help="Find exported HF checkpoints within a training run"
    )
    checkpoints.add_argument("run_dir", type=Path)
    demo = commands.add_parser(
        "demo", help="Complete synthetic walkthrough; no models, GPUs or code execution"
    )
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    return result


def main():
    args = parser().parse_args()
    try:
        root = args.root.resolve() if args.root else root_directory()
        command = args.command
        if command == "bootstrap":
            print(json.dumps(bootstrap(root), indent=2))
        elif command == "doctor":
            source = verify_checkout(root)
            print(
                json.dumps(
                    {
                        "python": sys.version.split()[0],
                        "uv": shutil.which("uv"),
                        "docker": shutil.which("docker"),
                        "trainer": source,
                        "trainer_environment": (root / "vendor/open-instruct/.venv").exists(),
                        "prepared_code": (root / "data/code/manifest.json").exists(),
                        "note": "No GPU queried and no model weights loaded",
                    },
                    indent=2,
                )
            )
        elif command == "validate":
            source = verify_checkout(root)
            for path in sorted((root / "configs/experiments").glob("*.toml")):
                config = resolve_config(root, path, hardware=args.hardware)
                plan = build_plan(root, config, root / "runs/validation-only")
                validate_flags(source["path"], plan["trainer"], method=plan["method"])
                template_text(source["path"], plan["trainer"]["chat_template_name"])
                print(f"Valid: {path.name} ({len(plan['trainer'])} trainer flags)")
        elif command == "prepare-code":
            options = {
                "unsupported_tests": args.unsupported_tests,
                "conflicting_prompts": args.conflicting_prompts,
                "cohort": args.cohort,
                "seed": args.seed,
                "validation_fraction": args.validation_fraction,
                "test_fraction": args.test_fraction,
            }
            manifest = (
                prepare_code_file(args.input, args.output, **options)
                if args.input
                else prepare_hub_code(root, args.output, **options)
            )
            print(json.dumps(manifest, indent=2))
        elif command == "prepare-benchmark":
            output = args.output or root / f"data/eval/{args.name}.jsonl"
            print(json.dumps(prepare_benchmark(root, args.name, output), indent=2))
        elif command == "filter-lengths":
            print(
                json.dumps(
                    filter_lengths(
                        root, args.input, args.output, args.max_prompt_tokens, args.max_total_tokens
                    ),
                    indent=2,
                )
            )
        elif command == "plan":
            config = resolve_config(root, args.experiment, args.hardware, args.seed)
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            run = (
                args.run_dir
                or root / "runs" / f"{config['experiment']['name']}-s{config['trainer']['seed']}-{timestamp}"
            )
            plan = build_plan(root, config, run)
            validate_flags(verify_checkout(root)["path"], plan["trainer"], method=plan["method"])
            save_plan(plan)
            print(f"Saved {run / 'plan.json'}\n{plan['shell_command']}")
        elif command == "launch":
            if args.execute:
                execute_plan(args.plan)
            else:
                print(json.loads(args.plan.read_text())["shell_command"])
                print("Plan only. --execute starts training.")
        elif command == "generate":
            lock = json.loads((root / "configs/sources.lock.json").read_text())
            model = lock["models"].get(args.model, {"id": args.model, "revision": args.revision})
            revision = args.revision or model.get("revision")
            if not Path(model["id"]).exists() and not revision:
                raise ValueError("A Hub model needs an explicit --revision (prefer a commit SHA)")
            arguments = [
                "--root",
                str(root),
                "--benchmark",
                str(args.benchmark.resolve()),
                "--model",
                model["id"],
                "--output",
                str(args.output.resolve()),
                "--evaluation-config",
                str(args.config.resolve()),
            ]
            if revision:
                arguments += ["--revision", revision]
            upstream_command(root, "rl_zero.generation", arguments, args.execute, args.cuda_group)
        elif command == "grade":
            if args.upstream:
                arguments = [
                    "--root",
                    str(root),
                    "--benchmark",
                    str(args.benchmark.resolve()),
                    "--predictions",
                    str(args.predictions.resolve()),
                    "--output",
                    str(args.output.resolve()),
                    "--k",
                    str(args.k),
                    "--api-url",
                    args.api_url,
                ]
                upstream_command(root, "rl_zero.grade_entry", arguments, execute=True)
            else:
                result = grade_predictions(
                    args.benchmark, args.predictions, args.output, args.k, api_url=args.api_url
                )
                print(f"pass@{args.k}: {result['score']:.4f}")
        elif command == "compare":
            print(json.dumps(write_comparison(args.before, args.after, args.output), indent=2))
        elif command == "dashboard":
            print(dashboard(args.run_dir, args.output or args.run_dir / "diagnostics.html"))
        elif command == "verifier":
            serve(args.host, args.port, args.image, args.trace_path, args.workers)
        elif command == "checkpoints":
            paths = [
                str(p.parent.resolve())
                for p in args.run_dir.rglob("config.json")
                if list(p.parent.glob("*.safetensors")) or list(p.parent.glob("pytorch_model*.bin"))
            ]
            for path in sorted(paths):
                print(path)
            if not paths:
                print("No exported HF weights found. Optimizer-state checkpoints need an export first.")
        elif command == "demo":
            print(run_demo(root, args.output))
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"rl-zero: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
