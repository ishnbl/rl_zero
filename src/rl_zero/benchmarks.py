"""Pinned benchmark snapshots with explicit local prompting/grading contracts."""

import json
from pathlib import Path

from rl_zero.data import prepare_code
from rl_zero.io import file_hash, write_json, write_jsonl


def load_hub_dataset(source):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ValueError("Install data support: python -m pip install -e '.[data]'") from error
    return load_dataset(
        source["id"], source.get("config"), split=source["split"], revision=source["revision"]
    )


def prepare_hub_code(root, output, **kwargs):
    lock = json.loads((Path(root) / "configs/sources.lock.json").read_text())
    source = lock["code_dataset"]
    return prepare_code(load_hub_dataset(source), output, source, **kwargs)


def normalize_benchmark(name, row, index):
    if name == "math500":
        return {
            "id": f"math500/{index}",
            "domain": "math",
            "grader": "math",
            "messages": [{"role": "user", "content": row["problem"]}],
            "answer": row["answer"],
        }
    if name == "gsm8k":
        return {
            "id": f"gsm8k/{index}",
            "domain": "math",
            "grader": "gsm8k",
            "messages": [{"role": "user", "content": row["question"]}],
            "answer": row["answer"].rsplit("####", 1)[-1].strip(),
        }
    if name == "mbpp":
        imports = "\n".join(row.get("test_imports", []))
        tests = [(imports + "\n" + test).strip() for test in row["test_list"]]
        prompt = row.get("prompt", row.get("text"))
        return {
            "id": f"mbpp/{row['task_id']}",
            "domain": "code",
            "tests": tests,
            "messages": [{"role": "user", "content": prompt}],
        }
    if name == "humaneval":
        return {
            "id": row["task_id"],
            "domain": "code",
            "messages": [
                {
                    "role": "user",
                    "content": "Write the complete Python implementation "
                    "for this function, including its signature:\n\n" + row["prompt"],
                }
            ],
            "tests": [row["test"] + f"\ncheck({row['entry_point']})"],
        }
    raise ValueError(f"Unsupported benchmark: {name}")


def prepare_benchmark(root, name, output):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite benchmark snapshot: {output}")
    source = json.loads((Path(root) / "configs/sources.lock.json").read_text())["benchmarks"][name]
    rows = [normalize_benchmark(name, row, i) for i, row in enumerate(load_hub_dataset(source))]
    write_jsonl(output, rows)
    manifest = {
        "source": source,
        "rows": len(rows),
        "sha256": file_hash(output),
        "protocol": "Local full-implementation code prompting; not official EvalPlus/OLMES scores",
    }
    write_json(str(output) + ".manifest.json", manifest)
    return manifest
