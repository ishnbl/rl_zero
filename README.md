# RL Zero Lab

A small experiment layer for studying **code learning and math retention** in Olmo 3.
Open Instruct owns distributed training; this repository owns the experiment design,
data cohorts, reward service, diagnostics, and comparisons.

The two initializations are pinned revisions of:

- `allenai/Olmo-3-1025-7B` — the actual base ID in the Math checkpoint's metadata.
- `allenai/Olmo-3-7B-RL-Zero-Math` — the checkpoint you linked, rather than its newer 3.1 replacement.

**No GPU training or inference was run during setup.** The lightweight environment,
upstream checkout, prepared data, benchmark snapshots, and synthetic walkthrough
are available locally. Code verification supports Daytona without local Docker;
Daytona credentials/a snapshot and the GPU training environment still need setup
on the eventual training machine. See [Daytona setup](docs/daytona.md).

## Start here

From this directory:

```bash
source .venv/bin/activate
rl-zero doctor
rl-zero validate
python -m unittest discover -s tests -v
```

An offline walkthrough already lives at `runs/final-demo`. It contains saved launch
plans, synthetic predictions, paired retention reports, and an interactive
`diagnostics.html`. Every result in that directory is synthetic.

```bash
rl-zero demo --output runs/another-demo
```

For a fresh checkout, install the lightweight tools and fetch the trainer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,data,audit]'
rl-zero bootstrap
```

`uv.lock` locks the lightweight package's extras. Open Instruct has a separate
lock and requires Python 3.12. Keep those environments separate. With uv, the
equivalent local setup is `uv sync --frozen --extra dev --extra data --extra audit`.

## The study

Evaluate **both initializations first**, on every chosen benchmark. Then train:

| Config in `configs/experiments/` | Initialization | Method | Cohort |
|---|---|---|---|
| `base_code_full_rlvr.toml` | Base | RLVR | Broader code cohort |
| `math_code_full_rlvr.toml` | Math RL-Zero | RLVR | Broader code cohort |
| `base_code_rlvr.toml` | Base | RLVR | Matched, length-controlled |
| `math_code_rlvr.toml` | Math RL-Zero | RLVR | Matched, length-controlled |
| `base_code_sft.toml` | Base | SFT | Matched, length-controlled |
| `math_code_sft.toml` | Math RL-Zero | SFT | Matched, length-controlled |

The first pair answers the initial base→code versus math→code RLVR question.
The four matched arms support the SFT comparison. Compare each trained checkpoint
to **its own initialization** to measure forgetting. Also compare the two trained
initializations directly to measure final capability differences.

The matched arms share training IDs, prompt formatting, and token eligibility.
They use different learning objectives and compute budgets. Equal data or update
counts do not establish equal compute. See [the methodology](docs/methodology.md).

## Prepare data

These commands download datasets and tokenizers, never model weights:

```bash
rl-zero prepare-code
rl-zero filter-lengths
rl-zero prepare-code --cohort full_rlvr --unsupported-tests exclude \
  --conflicting-prompts exclude --output data/code_full

rl-zero prepare-benchmark math500
rl-zero prepare-benchmark gsm8k
rl-zero prepare-benchmark mbpp
rl-zero prepare-benchmark humaneval
```

The public Dolci snapshot contains 13,312 rows. Exactly 6,656 have reference
solutions. The matched cohort keeps these for a fair data-population comparison:
6,052 train / 304 validation / 300 test. Both tokenizers retain all 6,052 training
examples under the default limits. SFT targets are the supplied solutions wrapped
in a Python fence; no reasoning traces or new reference answers are invented.

The broader cohort includes solution-free stdin/stdout tasks. Some rows have
ambiguous argument-array encodings without a calling convention, and some prompt
duplicates have different test sets. The explicit `exclude` options above omit
those cases and write their IDs/reasons to `excluded.jsonl`. Without those options,
preparation stops for inspection. **`full_rlvr` is the broader eligible cohort,
not an assertion that every raw source row is executable or an exact release reproduction.**

Every cohort has a `manifest.json` with source revision, split seed, counts, and
file hashes. Splits depend on normalized prompt hashes rather than row order.
Validation is for run selection; keep test and external benchmarks for reporting.

Bring another code dataset through the same boundary:

```bash
rl-zero prepare-code --input /path/to/rows.jsonl --output data/custom
```

Rows need `id`, `prompt` (text or system/user messages), `ground_truth` (assertion
strings or stdin/stdout cases), and `solution` for the matched cohort. See
[data.py](src/rl_zero/data.py) and [rewards.py](src/rl_zero/rewards.py).

## Inspect and launch a training run

Recipes hold the algorithm; hardware profiles hold GPU layout. The default
single-node profile allocates four learner GPUs and four rollout GPUs for RLVR,
or eight ZeRO-3 processes for SFT. It is a starting layout, not a measured memory
fit guarantee for 18k-token contexts.

First install the upstream environment on the training machine, then set up the
remote verifier following [Daytona setup](docs/daytona.md):

```bash
uv sync --project vendor/open-instruct --frozen --no-default-groups --group dev --group cuda12
python -m pip install -e '.[daytona]'
rl-zero daytona-snapshot  # Plan only; the guide covers remote creation and credentials.
rl-zero verifier --backend daytona --trace-path runs/verifier.jsonl
```

The verifier stays running in its terminal. Daytona creates a fresh sandbox per
candidate with blocked outbound networking and unprivileged candidate processes.
The environment includes NumPy, SciPy and SymPy. It records snapshot/runner
identity and deletes each sandbox after execution. No local Docker is required.
The optional Docker path is `docker build -f docker/Dockerfile.verifier -t
rl-zero-verifier:local .`, then `rl-zero verifier --backend docker`.

In another terminal:

```bash
source .venv/bin/activate
rl-zero plan configs/experiments/base_code_full_rlvr.toml --run-dir runs/base-code-first
cat runs/base-code-first/plan.json
rl-zero launch runs/base-code-first/plan.json
```

The last command only prints the saved command. To actually train later:

```bash
rl-zero launch runs/base-code-first/plan.json --execute
```

Repeat with the math initialization and matched/SFT configs. `--seed 2` creates
another seed; `--hardware configs/hardware/my-machine.toml` changes the layout.
Keep the two members of a comparison on the same recipe and hardware profile.
See [the architecture and extension guide](docs/architecture.md) for overrides,
resume, checkpoints, and the exact upstream loss/gradient entrypoints.

Uploads, Beaker jobs, and Slack notifications are disabled. W&B runs offline;
local JSONL traces remain usable without a W&B account.

## Evaluate math and code, then compare

Generation prints its command unless `--execute` is present. For example, a
future baseline evaluation is:

```bash
rl-zero generate --model base --benchmark data/eval/math500.jsonl \
  --output runs/eval/base.math500.jsonl --execute
rl-zero grade --upstream --benchmark data/eval/math500.jsonl \
  --predictions runs/eval/base.math500.jsonl --output runs/eval/base.math500.score.json
```

Use `--model math` for the other initialization, then an exported checkpoint path:

```bash
rl-zero checkpoints runs/base-code-first
rl-zero generate --model /absolute/path/to/exported/checkpoint \
  --benchmark data/eval/math500.jsonl --output runs/eval/base-code.math500.jsonl --execute
rl-zero grade --upstream --benchmark data/eval/math500.jsonl \
  --predictions runs/eval/base-code.math500.jsonl --output runs/eval/base-code.math500.score.json
rl-zero compare --before runs/eval/base.math500.score.json \
  --after runs/eval/base-code.math500.score.json --output reports/base-code.math-retention.json
```

For code, use `data/code/test.eval.jsonl`, MBPP, or HumanEval, and grade through
the running verifier (`--upstream` is unnecessary for code). `--k 8`
computes pass@8 from the same eight saved completions; the default is pass@1.

The JSON and Markdown comparison reports show signed percentage-point changes,
paired task improvements/regressions, and bootstrap intervals. Missing samples,
changed benchmarks, or incompatible protocols stop the comparison.

These are local research protocols. MBPP/HumanEval use full-implementation
generation, without EvalPlus's augmented tests; math uses the pinned Open
Instruct verifiers. Do not present these scores as official OLMES/EvalPlus results.

## Follow rewards and gradients

```bash
rl-zero dashboard runs/base-code-first
```

Open the resulting `diagnostics.html` and choose a metric. The run retains:

- `plan.json`, `command.sh`, and `provenance.json`: resolved recipe, pins, hashes, source snapshot.
- `train.log`, `metrics.jsonl`: stdout plus locally mirrored native scalar metrics.
- `gradients.jsonl`: SFT DeepSpeed global norms; RLVR exposes native `optim/grad_norm` in metrics.
- `rollouts/`: upstream rollout/reward/advantage traces.
- `worker_hooks_*.json`: evidence that runtime adaptations installed in workers.
- `verifier_failure_*.json`: an infrastructure failure that invalidates the run.
- `checkpoints/`, `training_state/`: exported weights and optimizer-state artifacts.

The separately selected verifier trace records programs, tests, results, latency,
image identity, and unscaled fractional rewards. Use a distinct `--trace-path`
per experiment for clean attribution. Training multiplies that reward by 10;
evaluation counts a solution correct only when **all** its tests pass.

Read [the source/provenance notes](docs/upstream.md) before changing templates or
calling a run a reproduction. Read [verification status](docs/verification.md)
for exactly what was exercised locally.
