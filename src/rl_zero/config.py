"""TOML composition: recipe → hardware method settings → experiment overrides."""

import copy
import json
import re
import tomllib
from pathlib import Path


def load_toml(path):
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def resolve_config(root, path, hardware=None, seed=None):
    root = Path(root).resolve()
    experiment = load_toml(root / path)
    method = experiment["method"]
    if method not in {"rlvr", "sft"}:
        raise ValueError("method must be rlvr or sft")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment["name"]):
        raise ValueError("Experiment name must be a simple filename component")
    recipe = load_toml(root / experiment["recipe"])
    hardware_path = hardware or experiment["hardware"]
    machine = load_toml(root / hardware_path)
    lock = json.loads((root / "configs/sources.lock.json").read_text())
    flags = copy.deepcopy(recipe["trainer"])
    flags.update(machine.get(method, {}))
    flags.update(experiment.get("trainer", {}))
    flags["seed"] = experiment["seed"] if seed is None else seed
    model = lock["models"][experiment["initialization"]]
    flags.update(
        model_name_or_path=model["id"],
        model_revision=model["revision"],
        tokenizer_name_or_path=model["id"],
        tokenizer_revision=model["revision"],
    )
    environment = {
        **recipe.get("environment", {}),
        **machine.get("environment", {}),
        **experiment.get("environment", {}),
    }
    return {
        "experiment": experiment,
        "trainer": flags,
        "environment": environment,
        "runtime": machine["runtime"],
        "hardware_path": str(hardware_path),
        "sources": lock,
    }
