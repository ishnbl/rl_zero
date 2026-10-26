# Working on RL Zero Lab

- Keep planning, reporting, and ordinary checks independent of torch/CUDA imports.
- Do not launch GPU training or inference unless the user explicitly requests a run.
- Use the lightweight `.venv` here; GPU dependencies belong to `vendor/open-instruct/.venv`.
- Run `make check` after activating `.venv`; `rl-zero validate` checks pinned upstream arguments.
- Never turn a verifier outage into a negative model reward or silently drop missing generations.
- Preserve cohort manifests and audit exclusions. Change scientific protocols under a new identity.
- Keep native upstream source clean and pinned. Runtime adaptations live in `runtime_hooks.py`.
- Keep tests focused on scientific boundaries. Avoid tests that merely copy configuration values.
- No automatic model uploads, external job submissions, or messages to other people.
