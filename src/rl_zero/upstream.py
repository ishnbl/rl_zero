"""Fetch and inspect the exact training implementation without importing CUDA packages."""

import ast
import json
import subprocess
from pathlib import Path

from rl_zero.io import file_hash
from rl_zero.templates import LOCAL_TEMPLATES


def git(checkout, *arguments):
    return subprocess.check_output(["git", "-C", str(checkout), *arguments], text=True).strip()


def bootstrap(root):
    root = Path(root)
    lock = json.loads((root / "configs/sources.lock.json").read_text())["open_instruct"]
    checkout = root / "vendor/open-instruct"
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", "--filter=blob:none", lock["url"], str(checkout)], check=True
        )
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", lock["commit"]], check=True)
    return verify_checkout(root)


def verify_checkout(root):
    root = Path(root)
    checkout = root / "vendor/open-instruct"
    pin = json.loads((root / "configs/sources.lock.json").read_text())["open_instruct"]["commit"]
    if not checkout.exists():
        raise ValueError("Missing trainer checkout; run rl-zero bootstrap")
    if git(checkout, "rev-parse", "HEAD") != pin:
        raise ValueError("Trainer revision differs from configs/sources.lock.json")
    if git(checkout, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Trainer has tracked edits; commit and update the lock before launching")
    return {"commit": pin, "uv_lock_sha256": file_hash(checkout / "uv.lock"), "path": str(checkout)}


def argument_fields(checkout, method):
    """Inspect dataclass annotations statically; importing the trainer would load CUDA."""
    selected = {"dataset_transformation.py": {"TokenizerConfig"}}
    if method == "sft":
        selected["finetune.py"] = {"FlatArguments"}
    elif method == "rlvr":
        selected.update(
            {
                "grpo_utils.py": {"GRPOExperimentConfig"},
                "olmo_core_utils.py": {
                    "ExperimentConfig",
                    "TrainingConfig",
                    "LoggingConfig",
                    "CheckpointConfig",
                },
                "model_utils.py": {"ModelConfig"},
                "data_loader.py": {"StreamingDataLoaderConfig", "VLLMConfig"},
                "environments/tools/utils.py": {"EnvsConfig"},
            }
        )
    else:
        raise ValueError("Unknown trainer method")
    names = set()
    for relative, classes in selected.items():
        path = Path(checkout) / "open_instruct" / relative
        tree = ast.parse(path.read_text())
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in classes):
            for node in cls.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names.add(node.target.id)
    return names


def validate_flags(checkout, flags, method="rlvr"):
    unknown = flags.keys() - argument_fields(checkout, method)
    if unknown:
        raise ValueError(f"Flags absent from pinned trainer dataclasses: {sorted(unknown)}")


def template_text(checkout, name):
    if name in LOCAL_TEMPLATES:
        return LOCAL_TEMPLATES[name]
    path = Path(checkout) / "open_instruct/dataset_transformation.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CHAT_TEMPLATES" for t in node.targets
        ):
            templates = ast.literal_eval(node.value)
            if name not in templates:
                raise ValueError(f"Unknown pinned template: {name}")
            return templates[name]
    raise ValueError("Cannot find CHAT_TEMPLATES in pinned source")
