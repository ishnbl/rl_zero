# Read and extend the experiment layer

The package has no runtime dependency for planning, basic reports, the verifier
gateway, or the synthetic demo. Dataset loading and tokenizer auditing are lazy
imports. Training and model inference run in Open Instruct's separate environment.

| Module | Responsibility |
|---|---|
| `config.py` | Recipe → hardware → experiment overrides |
| `upstream.py` | Source pin checks, static flag names, template extraction |
| `data.py`, `benchmarks.py` | Schema normalization, cohorts, splits and snapshots |
| `length_control.py` | Tokenizer-only common RLVR/SFT training eligibility |
| `launch.py`, `train_entry.py` | Reviewable commands, provenance, run lifecycle |
| `runtime_hooks.py` | Shared templates and fail-closed verifier in Ray workers |
| `rewards.py`, `verifier_client.py` | Test decoding and transparent reward semantics |
| `verifier.py`, `docker/runner.py` | HTTP API and isolated code execution |
| `instrumentation.py`, `diagnostics.py` | Local metrics and interactive viewer |
| `generation.py`, `evaluation.py` | Saved model completions and task grading |
| `report.py`, `metrics.py` | Paired capability/retention statistics |

## Follow one RLVR update

Open Instruct samples a group of responses, asks its CodeVerifier for fractional
execution rewards, scales rewards, computes centered advantages, and runs the
policy update. The local wrapper installs a replacement code-verifier call and
a shared prompt template in the driver and via Ray's worker setup hook.
It leaves loss computation and distributed optimization with upstream.

Useful upstream reading locations within `vendor/open-instruct/open_instruct/`:

- `data_loader.py`, `DataPreparationActor`: grouped rewards, filtering,
  advantages, and rollout trace serialization.
- `grpo_utils.py`, `compute_policy_loss`: DAPO/CISPO objectives, clipping,
  importance-ratio corrections, and KL.
- `grpo_fast.py`, `PolicyTrainerRayProcess.step`: backward, DeepSpeed step,
  global gradient norm, and per-update metrics.
- `ground_truth_utils.py`, `CodeVerifier`: original reward contract.
- `finetune.py`, `main`: assistant-only SFT updates and Accelerate integration.

`instrumentation.py` observes W&B Run.log so Accelerate and RLVR metrics remain
available locally. SFT additionally observes DeepSpeed's global norm after a
step. Do not interpret a single ZeRO shard's parameter gradients as the full
model gradient. Per-layer gradient capture would require explicit distributed
gather/reduction and is deliberately not claimed here.

## Add an experiment

Copy one short experiment TOML. Set `initialization`, `method`, `data_dir`, and
the recipe/hardware paths. An optional `[trainer]` table overrides upstream flags;
an `[environment]` table overrides environment settings. Model/tokenizer revisions
come from `configs/sources.lock.json`. Add a new model to that lock to introduce
another initialization.

Keep local publication flags explicitly false. To use online W&B deliberately,
set `WANDB_MODE` and your `wandb_entity` in a new recipe, preserving the old recipe
for existing comparisons. Training plans and runs refuse to reuse an existing
artifact directory.

Hardware profiles split `[runtime]` launcher settings from `[rlvr]` and `[sft]`
trainer flags. For multi-node work, provide shared paths and an externally managed
Ray cluster, and adapt the launch environment. The shipped profile is single-node;
there is no hidden cluster submission layer.

## Resume and checkpoint evaluation

Create a new run plan to preserve the original run's provenance. For SFT, put
`resume_from_checkpoint = "/absolute/path/to/completed/step_..."` in the new
experiment's `[trainer]` overrides. Use an optimizer-state checkpoint marked
complete by upstream, not merely an exported HF weight folder.

For RLVR, the pinned trainer loads state from `checkpoint_state_dir` automatically.
Use a new experiment's `[trainer]` overrides with that path pointing at the previous
state directory. The launcher normally assigns a fresh state directory; it preserves
an explicit override when planning. This path is also the future save destination;
copy the completed state to a new directory before resuming if preserving the
original state is important. Resume compatibility across GPU layouts is not
verified by this repository.

`rl-zero checkpoints RUN_DIR` finds directories with a config and weight files.
Open Instruct nests RLVR exports under its generated run name. SFT intermediate
checkpoints can be optimizer-only; use the final HF export, or the pinned
trainer's conversion path, for inference. The inventory command does not imply
that every optimizer checkpoint is directly loadable by vLLM.

## Failure artifacts and extension boundaries

The code service logs every attempted request. Infrastructure failures are
distinct from valid zero-reward responses. `runtime_hooks.py` writes failure
markers, raises the error through Ray, and the launcher invalidates the run.
The worker setup hook is an explicit runtime adaptation; inspect its receipts
on the first training run. No tracked file in the upstream checkout is patched.

The Docker image includes standard Python, NumPy, SciPy and SymPy. It cannot
verify arbitrary third-party APIs or services. Extend the image with pinned
dependencies when the dataset requires them, and keep its image ID in results.
For high throughput, implement a compatible external service with isolated
workers and matching `/health`, `/test_program` and `/test_program_stdio` contracts.
