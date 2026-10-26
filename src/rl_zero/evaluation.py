"""Grade saved completions. Missing outputs and verifier outages invalidate the run."""

import json
import urllib.request
from pathlib import Path

from rl_zero.io import file_hash, object_hash, read_jsonl, write_json, write_jsonl
from rl_zero.metrics import pass_at_k
from rl_zero.rewards import execution_reward, extract_code
from rl_zero.verifier_client import request_execution


def code_score(completion, tests, api_url):
    result = request_execution(api_url, extract_code(completion), tests)
    reward = execution_reward(result["results"])
    return {"passed": int(all(result["results"])), "reward": reward, "tests": result}


def grade_predictions(
    benchmark, predictions, output, k=1, api_url="http://127.0.0.1:1234/test_program", math_grader=None
):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite evaluation: {output}")
    tasks = list(read_jsonl(benchmark))
    responses = list(read_jsonl(predictions))
    metadata = json.loads(Path(str(predictions) + ".meta.json").read_text())
    if metadata["benchmark_sha256"] != file_hash(benchmark):
        raise ValueError("Predictions belong to a different benchmark snapshot")
    if metadata.get("predictions_sha256") != file_hash(predictions):
        raise ValueError("Predictions changed after generation")
    by_id = {item["id"]: item for item in responses}
    if len(by_id) != len(responses) or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    if set(by_id) != {item["id"] for item in tasks} or not tasks:
        raise ValueError("Predictions must cover every benchmark task exactly once")
    domains = {task["domain"] for task in tasks}
    if len(domains) != 1:
        raise ValueError("Grade one domain per benchmark")
    domain = domains.pop()
    if domain == "code":
        health_url = api_url.rsplit("/", 1)[0] + "/health"
        with urllib.request.urlopen(health_url, timeout=10) as response:
            health = json.load(response)
        if not health.get("image_id"):
            raise ValueError("Code grader health must expose an image_id for reproducible grading")
        grader = {
            "kind": "docker-code-v1",
            "image_id": health["image_id"],
            "per_test_seconds": 1.0,
            "extraction_sha256": object_hash(extract_code.__code__.co_code.hex()),
        }
    else:
        grader = metadata["grader_source"]
    records, scores = [], {}
    for task in tasks:
        completions = by_id[task["id"]]["completions"]
        if len(completions) != metadata["protocol"]["samples"]:
            raise ValueError(f"Incorrect sample count for {task['id']}")
        graded = []
        for sample in completions:
            if sample.get("error"):
                raise ValueError(f"Generation failed for {task['id']}: {sample['error']}")
            if domain == "code":
                value = code_score(sample["text"], task["tests"], api_url)
            elif math_grader:
                value = {"passed": int(math_grader(sample["text"], task["answer"], task.get("grader")))}
            else:
                raise ValueError("Math grading needs the pinned upstream environment; use grade --upstream")
            graded.append({**sample, **value})
        scores[task["id"]] = pass_at_k(len(graded), sum(s["passed"] for s in graded), k)
        records.append({"id": task["id"], "completions": graded, "pass_at_k": scores[task["id"]]})
    result = {
        "domain": domain,
        "k": k,
        "tasks": scores,
        "score": sum(scores.values()) / len(scores),
        "benchmark_sha256": file_hash(benchmark),
        "protocol": metadata["protocol"],
        "model": metadata["model"],
        "grader": grader,
        "predictions_sha256": file_hash(predictions),
        "samples": sum(len(r["completions"]) for r in records),
        "truncated": sum(s.get("finish_reason") == "length" for r in records for s in r["completions"]),
    }
    # Save detailed evidence only after all tasks have been successfully graded.
    write_jsonl(output.with_suffix(".samples.jsonl"), records)
    write_json(output, result)
    return result


def protocol_for(config, domain, template):
    """Exclude hardware/model identity, include every setting defining a comparison."""
    return {
        key: config[key]
        for key in ("name", "template_mode", "temperature", "top_p", "max_tokens", "samples", "seed")
    } | {
        "domain": domain,
        "template_sha256": object_hash(template),
        "sampling": "vllm.SamplingParams; stop_token_ids=tokenizer EOS; default top_k/min_p",
    }
