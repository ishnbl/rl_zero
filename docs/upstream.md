# Upstream provenance and deliberate adaptations

The source lock was resolved on 2026-10-26. The Math model's card metadata names
`allenai/Olmo-3-1025-7B` as its base. Some human-readable tables use older aliases
that do not resolve. The public code dataset currently resolves as
`allenai/Dolci-RL-Zero-Code-7B`, with hyphens differing from older script names.

Primary references:

- [Math checkpoint and metadata](https://huggingface.co/allenai/Olmo-3-7B-RL-Zero-Math/blob/main/README.md).
- [Public base checkpoint](https://huggingface.co/allenai/Olmo-3-1025-7B).
- [Public code dataset](https://huggingface.co/datasets/allenai/Dolci-RL-Zero-Code-7B).
- [Pinned current Code recipe](https://github.com/allenai/open-instruct/blob/3bb265641634d8ebe274311271a0b0dfbdcdd81f/scripts/train/olmo3/7b_rlzero_code.sh).
- [Upstream recipe and template history](https://github.com/allenai/open-instruct/blob/3bb265641634d8ebe274311271a0b0dfbdcdd81f/scripts/train/olmo3/README.md).
- [Ray worker setup hook contract](https://docs.ray.io/en/latest/ray-core/api/doc/ray.runtime_env.RuntimeEnv.html).

Upstream records original RL-Zero run code at `d928a7c`; launch scripts were
published later. The original code run used the historical `olmo_thinker`
template; the released Math checkpoint uses historical `olmo_thinker_dapo`.
Current main recipes and templates differ, and the current Math recipe is the
later 3.1 recipe. Changing one template flag does not reproduce a historical run.

This repository pins current dense-model `grpo_fast.py` for a practical local
research adaptation. It does not claim original run equivalence. The local
experiment choices include a single-node resource layout, reduced episode
budget, held-out data, disabled external job/publishing behavior, an isolated
local code service, and fail-closed infrastructure handling.

`olmo_lab_code` preserves the current upstream code generation prompt for one
user message and adds a stable assistant/EOS rendering for SFT. The upstream
template emits the reminder only during generation, so its generation prefix
does not remain a literal prefix of an SFT conversation. The CPU contract check
verifies local prompt equivalence and the supervised boundary. The local template
is installed dynamically, saved with trained tokenizers, and hashed in provenance.

Controlled evaluation fixes one template per domain across all models. Native
evaluation is available as a separate protocol. Template hashes and generation
package versions participate in comparison compatibility checks; grader identity
includes the pinned Open Instruct source or the Docker image ID.

Update the source lock intentionally. `bootstrap` will not replace a mismatched
existing checkout. Source verification rejects tracked upstream edits. Rerun
static validation and CPU contracts after a pin change, then perform an explicit
small GPU smoke run on the target hardware before a large study.
