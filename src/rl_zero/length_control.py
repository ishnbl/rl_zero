"""Shared train-population filtering using tokenizers only, never model weights."""

import importlib.metadata
import json
from pathlib import Path

from rl_zero.data import verify_prepared
from rl_zero.io import file_hash, object_hash, read_jsonl, write_json, write_jsonl
from rl_zero.templates import CODE_TEMPLATE


def filter_lengths(root, input_dir, output_dir, max_prompt_tokens=2048, max_total_tokens=18432):
    from transformers import AutoTokenizer

    root, input_dir, output_dir = Path(root), Path(input_dir), Path(output_dir)
    library_version = importlib.metadata.version("transformers")
    hub_version = importlib.metadata.version("huggingface-hub")
    if library_version != "5.4.0" or hub_version != "1.8.0":
        raise ValueError(
            "Length auditing needs trainer pins transformers==5.4.0 and huggingface-hub==1.8.0; "
            "install the audit extra"
        )
    manifest = verify_prepared(input_dir)
    if manifest["cohort"] != "matched":
        raise ValueError("Shared RLVR/SFT length control requires the matched cohort")
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite length-controlled data: {output_dir}")
    sources = json.loads((root / "configs/sources.lock.json").read_text())["models"]
    tokenizers = [
        AutoTokenizer.from_pretrained(m["id"], revision=m["revision"], trust_remote_code=False)
        for m in sources.values()
    ]
    sft = {r["prompt_hash"]: r for r in read_jsonl(input_dir / "train.sft.jsonl")}
    kept, excluded = set(), []
    longest_prompt, longest_sft = 0, 0
    for row in read_jsonl(input_dir / "train.rlvr.jsonl"):
        key = row["prompt_hash"]
        prompt_lengths = [
            len(
                t.apply_chat_template(
                    row["messages"],
                    chat_template=CODE_TEMPLATE,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
            )
            for t in tokenizers
        ]
        full_lengths = [
            len(
                t.apply_chat_template(
                    sft[key]["messages"],
                    chat_template=CODE_TEMPLATE,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=False,
                )
            )
            for t in tokenizers
        ]
        longest_prompt = max(longest_prompt, max(prompt_lengths))
        longest_sft = max(longest_sft, max(full_lengths))
        if max(prompt_lengths) <= max_prompt_tokens and max(full_lengths) <= max_total_tokens:
            kept.add(key)
        else:
            excluded.append(
                {
                    "id": row["id"],
                    "prompt_hash": key,
                    "prompt_lengths": prompt_lengths,
                    "sft_lengths": full_lengths,
                }
            )
    if not kept:
        raise ValueError("No training examples survive the shared token limits")
    files = {}
    for name in manifest["files"]:
        rows = list(read_jsonl(input_dir / name))
        if name.startswith("train."):
            rows = [r for r in rows if r["prompt_hash"] in kept]
        write_jsonl(output_dir / name, rows)
        files[name] = {"sha256": file_hash(output_dir / name), "rows": len(rows)}
    write_jsonl(output_dir / "length_excluded.jsonl", excluded)
    result = {
        **manifest,
        "files": files,
        "parent_manifest_sha256": file_hash(input_dir / "manifest.json"),
        "length_control": {
            "transformers_version": library_version,
            "huggingface_hub_version": hub_version,
            "models": sources,
            "template_sha256": object_hash(CODE_TEMPLATE),
            "max_prompt_tokens": max_prompt_tokens,
            "max_total_tokens": max_total_tokens,
            "observed_max_prompt_tokens": longest_prompt,
            "observed_max_sft_tokens": longest_sft,
            "kept_train": len(kept),
            "excluded_train": len(excluded),
            "heldout_policy": "validation and test kept unchanged",
        },
    }
    write_json(output_dir / "manifest.json", result)
    return result["length_control"]
