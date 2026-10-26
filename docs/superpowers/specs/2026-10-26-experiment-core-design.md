# Code transfer and math retention experiments

Build a readable experiment layer for two initializations (Olmo 3 base and the
released Math RL-Zero checkpoint) and two code-training methods (RLVR and SFT).
Measure each trained checkpoint against its own initialization and compare
training methods using identical held-out examples and generation settings.

## Approach

Use pinned Open Instruct for distributed training. A standalone trainer would
make the loop shorter to read but would create another RL implementation to
validate. A fork would make changes harder to audit against upstream. The thin
adapter is the implementation choice here; the user requested reusable tooling
and explicitly authorized repository setup, with no GPU runs.

Python's standard library powers configuration, launch planning, reporting, the
verifier gateway, and offline examples. Optional datasets dependencies support
Hub downloads. Training has its own upstream Python 3.12 environment, separate
from the lightweight local environment. TOML recipes compose with hardware and
experiment overrides. Nothing imports torch for planning or reporting.

## Scientific controls

- Pin model, data, and trainer revisions; retain resolved config, commands,
  source hashes, environment versions, logs, and run status.
- Normalize Dolci's prompt, solution and nested test schema once. Split by a
  stable prompt hash, deduplicate prompts, and use the same training IDs for
  RLVR and SFT. Keep validation and test examples out of training.
- Evaluate both initializations before training. Use shared prompts/templates
  for controlled comparisons, with checkpoint-native templates as a separate
  protocol. Never silently pool protocols or substitute generation failures.
- Distinguish fractional unit-test training reward from all-tests-pass task
  success; compute unbiased pass@k. Report signed math deltas and paired
  regressions/improvements with uncertainty.
- SFT uses supplied reference solutions, not matched RL compute. Record token
  and episode budgets; do not claim causal method comparisons from equal steps.

## Components

`config.py` composes validated configs; `upstream.py` pins/checks sources;
`data.py` handles normalization and split manifests; `launch.py` writes and
executes plans; `rewards.py` defines transparent execution rewards;
`verifier.py` serves the Open Instruct code API using restricted Docker workers;
`instrumentation.py` mirrors native tracking metrics locally; `evaluation.py`
generates/scorers predictions; `report.py` compares paired results; `cli.py`
provides explicit commands; `demo.py` exercises the workflow without models.

## Observability and limits

Retain rollout traces, per-test results and latency, reward, group statistics,
native global gradient norm, loss, clipping and KL. Distributed global gradient
norm comes from DeepSpeed rather than misleading per-shard parameter hooks.
Document exact upstream entrypoints where humans can change loss and rewards.

Use the current pinned Code recipe as an adaptation, not an asserted reproduction
of the original release: upstream documents template and recipe drift. Hardware
defaults are editable planning examples, not a fit guarantee. Generated code is
only executed inside Docker workers, never by the host Python interpreter.
No model downloads, GPU execution, paid services, or publishing during setup.

## Verification

Keep a small CPU suite around scientifically dangerous boundaries: data leakage,
reward semantics, pass@k, incompatible evaluation protocols, launch safety and
upstream argument drift. Exercise the CLI with a complete offline demo and
validate generated arguments against the pinned source without importing it.
