# Daytona verifier implementation plan

**Goal:** Run code rewards in remote Daytona containers when local Docker is unavailable.

**Design:** Preserve the existing HTTP contract and reward runner. A small backend
interface selects Docker or Daytona. Daytona uses a prebuilt snapshot, creates a
private, network-blocked ephemeral sandbox per candidate, uploads the runner and
JSON payload, executes it, validates results, and deletes the sandbox in `finally`.
Snapshot builds happen remotely through a separate explicit command, with dry-run
as its default. No cloud provisioning or GPU work during implementation.

**Alternatives:** Per-candidate remote image builds add latency; a shared mutable
sandbox risks cross-candidate contamination. A snapshot plus disposable sandboxes
preserves a stable environment and a clear lifecycle. This is a scoped extension
to the repository setup already authorized by the user.

- [x] Add focused offline lifecycle/error checks, using the pinned SDK types.
- [x] Extract the shared runner and implement backend selection and Daytona configuration.
- [x] Add remote snapshot planning/building and backend-aware evaluation provenance.
- [x] Update setup docs, lock dependencies, verify locally, and review the change.

**Verification limits:** SDK/API contracts and offline lifecycle checks can run
here. Real sandbox execution needs user credentials and an explicitly requested
cloud run. Do not claim remote execution was tested.

Review addressed expected-output visibility, creation-timeout cleanup, and immutable runner identity. Seven additional offline checks pass, including the local HTTP/evaluation path. SDK models were verified against installed 0.226.1 source. No live API calls or cloud provisioning were performed.
