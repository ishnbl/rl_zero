"""Explicit inference entrypoint. Imported only in the trainer environment."""

import argparse
import importlib.metadata
from pathlib import Path

from rl_zero.config import load_toml
from rl_zero.evaluation import protocol_for
from rl_zero.io import file_hash, read_jsonl, write_json, write_jsonl
from rl_zero.upstream import template_text, verify_checkout


def generate(root, benchmark, model, revision, output, evaluation_config):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    root, output = Path(root), Path(output)
    if output.exists() or Path(str(output) + ".meta.json").exists():
        raise ValueError(f"Refusing to overwrite predictions: {output}")
    source = verify_checkout(root)
    config = load_toml(evaluation_config)
    tasks = list(read_jsonl(benchmark))
    if not tasks or len({t["domain"] for t in tasks}) != 1:
        raise ValueError("Generation needs a nonempty single-domain benchmark")
    domain = tasks[0]["domain"]
    if config["template_mode"] not in {"controlled", "native"}:
        raise ValueError("template_mode must be controlled or native")
    packages = {name: importlib.metadata.version(name) for name in ("vllm", "transformers", "torch")}
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    template = (
        template_text(source["path"], config[f"{domain}_template"])
        if config["template_mode"] == "controlled"
        else tokenizer.chat_template
    )
    if not template:
        raise ValueError("No chat template; choose the controlled evaluation protocol")
    prompts = [
        tokenizer.apply_chat_template(
            t["messages"], chat_template=template, tokenize=False, add_generation_prompt=True
        )
        for t in tasks
    ]
    # Long prompts are never silently truncated. vLLM raises when the context does not fit.
    engine = LLM(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        trust_remote_code=False,
        tensor_parallel_size=config["tensor_parallel_size"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        seed=config["seed"],
        max_model_len=config["max_tokens"] + 4096,
    )
    sampling = SamplingParams(
        n=config["samples"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
        seed=config["seed"],
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [],
    )
    outputs = engine.generate(prompts, sampling)
    records = [
        {
            "id": task["id"],
            "completions": [
                {"text": o.text, "finish_reason": o.finish_reason, "tokens": len(o.token_ids)}
                for o in generated.outputs
            ],
        }
        for task, generated in zip(tasks, outputs, strict=True)
    ]
    identity = {"id": model, "revision": revision}
    if Path(model).is_dir():
        identity["files_sha256"] = {
            p.name: file_hash(p)
            for p in sorted(Path(model).iterdir())
            if p.is_file() and p.suffix in {".json", ".jinja", ".safetensors", ".bin"}
        }
    protocol = protocol_for(config, domain, template)
    protocol["packages"] = packages
    protocol["eos_token_id"] = tokenizer.eos_token_id
    protocol["generation_code_sha256"] = file_hash(__file__)
    write_jsonl(output, records)
    write_json(
        str(output) + ".meta.json",
        {
            "model": identity,
            "benchmark_sha256": file_hash(benchmark),
            "predictions_sha256": file_hash(output),
            "protocol": protocol,
            "grader_source": "open-instruct:" + source["commit"],
            "packages": packages,
            "template": template,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "benchmark", "model", "output", "evaluation-config"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--revision")
    args = parser.parse_args()
    generate(args.root, args.benchmark, args.model, args.revision, args.output, args.evaluation_config)


if __name__ == "__main__":
    main()
