"""Compare checkpoints only when examples, decoding, template, and grader agree."""

import json
from pathlib import Path

from rl_zero.io import write_json
from rl_zero.metrics import paired_changes


def compare_results(before, after):
    for field in ("protocol", "benchmark_sha256", "domain", "grader", "k"):
        if before[field] != after[field]:
            raise ValueError(f"Cannot compare evaluations with different {field}")
    return {"domain": before["domain"], "k": before["k"], **paired_changes(before["tasks"], after["tasks"])}


def write_comparison(before_path, after_path, output):
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())
    result = compare_results(before, after)
    result.update(before=str(before_path), after=str(after_path))
    write_json(output, result)
    lo, hi = result["delta_ci95_pp"]
    text = (
        f"# {result['domain']} retention: pass@{result['k']}\n\n"
        f"{result['n']} paired tasks. Scores are percentages; deltas are percentage points.\n\n"
        "| Before | After | Change | Paired bootstrap 95% CI |\n"
        "|---:|---:|---:|---:|\n"
        f"| {result['before_pct']:.2f} | {result['after_pct']:.2f} | "
        f"{result['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |\n\n"
        f"Tasks with lower estimates: {result['regressed']}; higher: {result['improved']}; "
        f"unchanged: {result['unchanged']}. Negative math deltas indicate net forgetting.\n\n"
        "The interval resamples tasks; it does not capture training-seed uncertainty.\n"
    )
    Path(output).with_suffix(".md").write_text(text)
    return result
