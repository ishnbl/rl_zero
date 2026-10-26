"""An offline walkthrough. All model outputs/metrics are explicitly synthetic."""

from pathlib import Path

from rl_zero.config import resolve_config
from rl_zero.data import prepare_code
from rl_zero.diagnostics import dashboard
from rl_zero.evaluation import grade_predictions
from rl_zero.instrumentation import append_metrics
from rl_zero.io import file_hash, write_json, write_jsonl
from rl_zero.launch import build_plan, save_plan
from rl_zero.report import write_comparison


def run_demo(root, output):
    root, output = Path(root), Path(output).resolve()
    if output.exists():
        raise ValueError(f"Demo output already exists: {output}; choose a new directory")
    rows = [
        {
            "id": f"synthetic-{i}",
            "prompt": f"Implement add_{i}(a, b), returning a+b.",
            "solution": f"def add_{i}(a, b): return a + b",
            "ground_truth": [f'["assert add_{i}(2, 3) == 5"]'],
        }
        for i in range(100)
    ]
    prepare_code(
        rows, output / "data", {"kind": "synthetic-demo"}, validation_fraction=0.2, test_fraction=0.2
    )
    for config_path in sorted((root / "configs/experiments").glob("*.toml")):
        config = resolve_config(root, config_path)
        config["experiment"]["data_dir"] = str(output / "data")
        save_plan(build_plan(root, config, output / config["experiment"]["name"]))
    benchmark = output / "math.synthetic.jsonl"
    tasks = [
        {
            "id": f"demo-math-{i}",
            "domain": "math",
            "grader": "synthetic-exact",
            "messages": [{"role": "user", "content": f"What is {i} + 1?"}],
            "answer": str(i + 1),
        }
        for i in range(12)
    ]
    write_jsonl(benchmark, tasks)
    protocol = {"name": "SYNTHETIC-demo-only", "samples": 4, "temperature": 1, "seed": 42}
    for label, offset in (
        ("base", 0),
        ("math", 2),
        ("base_code_rlvr", 1),
        ("math_code_rlvr", 1),
        ("base_code_sft", -1),
        ("math_code_sft", 0),
    ):
        predictions = output / f"{label}.predictions.jsonl"
        write_jsonl(
            predictions,
            [
                {
                    "id": t["id"],
                    "completions": [
                        {
                            "text": t["answer"] if (i + j + offset) % 5 < 3 else "wrong",
                            "finish_reason": "stop",
                        }
                        for j in range(4)
                    ],
                }
                for i, t in enumerate(tasks)
            ],
        )
        write_json(
            str(predictions) + ".meta.json",
            {
                "benchmark_sha256": file_hash(benchmark),
                "predictions_sha256": file_hash(predictions),
                "protocol": protocol,
                "model": {"id": label, "revision": "SYNTHETIC"},
                "grader_source": "synthetic-exact",
            },
        )
        grade_predictions(
            benchmark,
            predictions,
            output / f"{label}.score.json",
            k=1,
            math_grader=lambda text, answer, _: text == answer,
        )
    for arm in ("base_code_rlvr", "math_code_rlvr", "base_code_sft", "math_code_sft"):
        baseline = "math" if arm.startswith("math") else "base"
        write_comparison(
            output / f"{baseline}.score.json", output / f"{arm}.score.json", output / f"{arm}.retention.json"
        )
    run = output / "base_code_rlvr"
    for step in range(20):
        append_metrics(
            run,
            {
                "SYNTHETIC/grad_norm": 2 / (step + 1),
                "SYNTHETIC/reward": step / 20,
                "SYNTHETIC/loss": 1 / (step + 1),
            },
            step,
        )
    dashboard(run, output / "diagnostics.html")
    (output / "README.txt").write_text(
        "SYNTHETIC WALKTHROUGH ONLY. No model inference, code execution, or training.\n"
    )
    return str(output)
