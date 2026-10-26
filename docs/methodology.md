# Experimental controls

Measure four quantities separately: final code capability, final math capability,
change from a checkpoint's own initialization, and differences between training
methods. A higher final score does not establish lower forgetting: Math RL-Zero
starts from a different capability level.

The minimum study has two baseline checkpoints and four trained checkpoints.
The broader eligible RLVR cohort is a separate experiment from the matched
method comparison. Never mix those cohorts into a single RLVR-versus-SFT claim.

## Training population and budget

Matched preparation selects source rows with supplied reference solutions.
Prompt-hash deduplication and deterministic partitioning prevent a repeated
prompt from crossing partitions. The length-control command checks both pinned
initialization tokenizers with the shared training template. It admits training
examples only when prompts fit 2,048 tokens and supervised conversations fit
18,432 tokens. Held-out populations stay unchanged.

Training source solutions are not independently verified during preparation;
data schema checks establish format, not solution correctness. Validate a sample
through your chosen code environment before a large SFT study.

The default RLVR budget is 25,600 response episodes: 32 unique prompts × 8
responses per rollout, nominally 100 learner updates. Dynamic filtering can
consume additional rollouts. The SFT budget is one epoch, with a global batch
of 8 processes × 1 example × 4 accumulation steps. These budgets are experiment
starting points, not compute-matched comparisons. Retain generated/trained token
counts, updates, wall time and source-cohort identity. Run multiple training seeds.

## Reward and objective

The code reward is `sum(test_passes) / number_of_tests`, set to zero below the
configured threshold. The default threshold is zero; empty test sets are invalid.
The upstream multiplier is 10. The current recipe uses the DAPO loss path, centered
group rewards, zero KL coefficient, one epoch per update, asymmetric clipping,
and truncated-response masking. See the resolved recipe for every explicit value.

The readable replacement verifier preserves valid-response fractional reward
semantics, while infrastructure failures propagate and invalidate the run. It
uses a configurable 900-second HTTP timeout for queueing and sequential tests.
The launcher watches failure artifacts and stops the process group. The first
real GPU smoke run should verify hook installation and failure shutdown behavior.

Docker execution is an adaptation: a fresh interpreter per test, an explicit
package environment, and process startup included in the time budget. Stdio
comparison strips lines and uses exact Decimal equality for numeric lines;
it does not guess function-call semantics for argument arrays. These differences
from the historical execution service can change rewards.

## Evaluation protocol

Freeze benchmark snapshots, template text, generation library versions,
temperature, top-p, token cap, sample count, and seed. The default controlled
protocol uses the same domain-specific template across all checkpoints. A separate
`template_mode = "native"` evaluation can measure checkpoint-native behavior;
its scores cannot be pooled with the controlled protocol. Base checkpoints may
not ship a usable native chat template.

Use pass@1 for an average single sample and pass@k for search with a stated sample
budget. The estimator is `1 - C(n-c,k) / C(n,k)`, where `c` counts complete task
successes, not mean unit-test rewards. Missing or failed generations invalidate
an evaluation. Truncated completions remain in the denominator and are graded;
the summary records their count. Change that policy only as a new protocol.

For math retention, report `after - before` in percentage points against the same
model's initialization. Negative values mean net regression on that benchmark.
Per-task decreases and increases are paired estimates; bootstrap confidence
intervals resample tasks, not training seeds, and do not establish significance
across training runs.

Use validation for selection and test for final reporting. Checkpoints saved at
different training stages can be compared under the same evaluation protocol;
choose selection rules before inspecting test results. Evaluate both domains
through the training trajectory, not just the final checkpoint.

Exact normalized prompt overlap was checked locally between the matched training
cohort and the prepared external benchmarks; none was found. This does not rule
out semantic overlap, paraphrases, shared underlying problems, or pretraining
contamination. Make that limitation explicit when interpreting benchmark gains.
