"""Normalize once; reuse identical prompt partitions in every training arm."""

import hashlib
import json
import re
from pathlib import Path

from rl_zero.io import file_hash, object_hash, read_jsonl, write_json, write_jsonl
from rl_zero.rewards import decode_tests


def normalize_messages(prompt):
    if isinstance(prompt, str):
        # Public Dolci stores a serialized user-role prefix in the prompt string.
        content = re.sub(r"^user:\s*", "", prompt.strip(), count=1)
        prompt = [{"role": "user", "content": content}]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("Expected prompt text or a nonempty chat message list")
    if any(
        m.get("role") not in {"system", "user"}
        or not isinstance(m.get("content"), str)
        or not m["content"].strip()
        for m in prompt
    ):
        raise ValueError("Training prompts must contain only nonempty system/user messages")
    if prompt[-1]["role"] != "user":
        raise ValueError("Prompt must end in a user message")
    return [{"role": m["role"], "content": m["content"].strip()} for m in prompt]


def normalize_code_row(row):
    messages = normalize_messages(row.get("messages", row.get("prompt")))
    canonical = [{"role": m["role"], "content": re.sub(r"\s+", " ", m["content"]).strip()} for m in messages]
    prompt_hash = object_hash(canonical)
    solution = row.get("solution", "")
    solution = solution if isinstance(solution, str) and solution.strip() else None
    tests = decode_tests(row["ground_truth"])
    return {
        "id": str(row.get("id", prompt_hash)),
        "prompt_hash": prompt_hash,
        "messages": messages,
        "solution": solution,
        "ground_truth": tests,
        "dataset": "code_stdio" if isinstance(tests[0], dict) else "code",
    }


def partition_code(
    rows, seed=42, validation_fraction=0.05, test_fraction=0.05, conflicting_prompts="error", exclusions=None
):
    if (
        not 0 < validation_fraction < 1
        or not 0 < test_fraction < 1
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError("Validation and test fractions must be positive and sum to less than 1")
    unique, conflicted = {}, set()
    for raw in rows:
        row = normalize_code_row(raw)
        key = row["prompt_hash"]
        if key in conflicted:
            if exclusions is not None:
                exclusions.append(
                    {"id": row["id"], "reason": "conflicting_duplicate_prompt", "prompt_hash": key}
                )
            continue
        if key in unique and row["ground_truth"] != unique[key]["ground_truth"]:
            if conflicting_prompts != "exclude":
                raise ValueError(
                    f"Conflicting tests for duplicate prompt {key}; curate this source or explicitly "
                    "use --conflicting-prompts exclude"
                )
            if exclusions is not None:
                exclusions.extend(
                    {"id": r["id"], "reason": "conflicting_duplicate_prompt", "prompt_hash": key}
                    for r in (unique[key], row)
                )
            del unique[key]
            conflicted.add(key)
            continue
        if key not in unique or row["id"] < unique[key]["id"]:
            unique[key] = row
    parts = {"train": [], "validation": [], "test": []}
    for key, row in sorted(unique.items()):
        bucket = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), 16) / 2**256
        split = (
            "test"
            if bucket < test_fraction
            else ("validation" if bucket < test_fraction + validation_fraction else "train")
        )
        parts[split].append(row)
    return parts


def prepare_code(
    rows,
    output,
    source,
    seed=42,
    validation_fraction=0.05,
    test_fraction=0.05,
    cohort="matched",
    unsupported_tests="error",
    conflicting_prompts="error",
):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite prepared data: {output}")
    if cohort not in {"matched", "full_rlvr"}:
        raise ValueError("cohort must be matched or full_rlvr")
    selected, excluded, source_count = [], [], 0
    for row in rows:
        source_count += 1
        if cohort == "matched" and not (isinstance(row.get("solution"), str) and row["solution"].strip()):
            excluded.append({"id": str(row.get("id")), "reason": "missing_reference_solution"})
        else:
            try:
                decode_tests(row["ground_truth"])
            except ValueError as error:
                if unsupported_tests != "exclude":
                    raise ValueError(
                        f"{row.get('id')}: {error}; inspect this schema or explicitly use "
                        "--unsupported-tests exclude (writes an audit file)"
                    ) from error
                excluded.append(
                    {"id": str(row.get("id")), "reason": "unsupported_test_schema", "detail": str(error)}
                )
                continue
            selected.append(row)
    prior_excluded = len(excluded)
    parts = partition_code(
        selected,
        seed,
        validation_fraction,
        test_fraction,
        conflicting_prompts=conflicting_prompts,
        exclusions=excluded,
    )
    conflict_exclusions = len(excluded) - prior_excluded
    if any(not rows for rows in parts.values()):
        raise ValueError("A partition is empty; use more examples or larger holdout fractions")
    files = {}
    for split, items in parts.items():
        rlvr = [
            {k: row[k] for k in ("id", "prompt_hash", "messages", "ground_truth", "dataset")} for row in items
        ]
        path = output / f"{split}.rlvr.jsonl"
        write_jsonl(path, rlvr)
        files[path.name] = {"sha256": file_hash(path), "rows": len(items)}
        if split == "train" and cohort == "matched":
            sft = [
                {
                    "id": r["id"],
                    "prompt_hash": r["prompt_hash"],
                    "messages": r["messages"]
                    + [{"role": "assistant", "content": "\n```python\n" + r["solution"].strip() + "\n```"}],
                }
                for r in items
            ]
            path = output / "train.sft.jsonl"
            write_jsonl(path, sft)
            files[path.name] = {"sha256": file_hash(path), "rows": len(items)}
        elif split != "train":
            path = output / f"{split}.eval.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "id": r["prompt_hash"],
                        "source_id": r["id"],
                        "domain": "code",
                        "test_kind": r["dataset"],
                        "messages": r["messages"],
                        "tests": r["ground_truth"],
                    }
                    for r in items
                ],
            )
            files[path.name] = {"sha256": file_hash(path), "rows": len(items)}
    manifest = {
        "cohort": cohort,
        "unsupported_tests_policy": unsupported_tests,
        "conflicting_prompts_policy": conflicting_prompts,
        "source_rows": source_count,
        "excluded_rows": len(excluded),
        "duplicates_removed": len(selected) - conflict_exclusions - sum(map(len, parts.values())),
        "source": source,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "deduplication": "normalized prompt SHA256",
        "files": files,
    }
    write_jsonl(output / "excluded.jsonl", excluded)
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_prepared(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        if file_hash(output / name) != info["sha256"]:
            raise ValueError(f"Data changed after preparation: {name}")
    return manifest


def prepare_code_file(path, output, **kwargs):
    return prepare_code(
        read_jsonl(path), output, {"path": str(Path(path).resolve()), "sha256": file_hash(path)}, **kwargs
    )
