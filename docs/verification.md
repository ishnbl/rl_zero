# Verification performed on 2026-10-26

The subsequent Daytona extension adds seven offline checks (17 total when the
optional SDK is installed), covering provider lifecycle, creation-timeout cleanup,
immutable runner bytes, expected-output isolation, result validation, local HTTP
grading and dry-run snapshot planning. Daytona SDK 0.226.1 request models were
constructed locally and checked against its installed source. No authenticated
Daytona requests, cloud provisioning or live remote execution were performed.
See [the Daytona runbook](daytona.md).

All work ran on CPUs. No model weights were downloaded, no GPU was queried, and
no training or model inference was launched. The local environment has no torch
installation. Dataset files and tokenizer files were downloaded intentionally.

## Checked

- Ten focused `unittest` contracts: nested/primitive-only compressed tests,
  fractional reward boundaries, prompt isolation, deterministic deduplication,
  pass@k, paired deltas, incompatible protocols, safe launch defaults, shared
  prompt/SFT boundaries, and verifier-error propagation.
- Ruff lint and formatting, plus Python compilation.
- All six experiment recipes against the selected pinned entrypoint's dataclass
  fields, with upstream imports avoided.
- End-to-end synthetic walkthrough: data manifests, six launch plans, saved
  completions, grading, paired reports and interactive diagnostics.
- Actual preparation of the matched and broader eligible Dolci cohorts; all
  produced file hashes verified against their manifests.
- Both pinned tokenizers under the trainer's Transformers 5.4.0 and Hub 1.8.0
  versions: 6,052 common training examples retained, longest prompt 712 tokens,
  longest supervised conversation 3,029 tokens. Explicit `return_dict=False`
  ensures these are token counts rather than BatchEncoding field counts.
- Pinned external snapshots: MATH-500 (500), GSM8K (1,319), sanitized MBPP (257),
  and HumanEval (164).
- Exact normalized prompt overlap between matched train and those external
  snapshots: zero. This check is not semantic decontamination.
- Ray 2.53.0 CPU setup-hook transport and propagation of verifier exceptions,
  using `scripts/check_ray_hook.py` with `num_gpus=0`.

The final artifacts are in `runs/final-demo/` and `runs/ready-plans/`. The latter
contains plans for the real prepared datasets; none has been executed.

## Actual curation counts

| Cohort | Train | Validation | Test | Source exclusions |
|---|---:|---:|---:|---:|
| Matched / length-controlled | 6,052 | 304 | 300 | 6,656 without references |
| Broader eligible RLVR | 11,592 | 603 | 587 | 415 ambiguous test schemas; 112 conflicting duplicate-prompt rows |

The broader cohort additionally removes three identical prompt duplicates.
The split hash keeps shared prompts in the same partition across these cohorts.

## Boundaries still requiring runtime validation

Docker is unavailable here, so the image build, container execution, resource
limits and stdout/stdin integration were not exercised. Review fixed a stderr
mixing issue: only stdout now participates in stdin/stdout answer comparison.
The image includes the pinned scientific Python packages, but arbitrary library
requirements still need a task-specific environment.

GPU memory fit, distributed training, real rollout throughput, full worker-hook
integration with the trainer modules, SFT gradient observers, model inference,
and optimizer-state resume have not been executed. The independent CPU Ray probe
checks hook transport and exception propagation, not the entire GPU trainer.

Before a large experiment, explicitly run a small smoke job on the target setup,
inspect worker hook receipts, test a verifier outage, check a few rewards by hand,
and evaluate both initializations. This repository provides that workflow but
does not report synthetic results as model evidence.

Re-run local verification with:

```bash
source .venv/bin/activate
make check
rl-zero validate
```

The optional Ray integration probe requires `python -m pip install ray==2.53.0`;
it was installed locally for verification but is not a core dependency.
