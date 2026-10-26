# Code rewards through Daytona

Daytona accepts Docker images and can build snapshots remotely, so this backend
does not need a local Docker daemon. The reward HTTP service still runs here;
only candidate execution happens in Daytona sandboxes.

Install the optional pinned SDK:

```bash
source .venv/bin/activate
python -m pip install -e '.[daytona]'
```

Set `DAYTONA_API_KEY` in your shell or a local `.env` (already git-ignored).
The SDK also supports `DAYTONA_API_URL` and `DAYTONA_TARGET` for a custom endpoint
or region. Credentials stay with the gateway; candidate environments receive
neither the API key nor the host environment.

## Build the environment remotely

The default config is [daytona.toml](../configs/verifiers/daytona.toml). It requests
a CPU-only snapshot with Python 3.12, NumPy, SciPy and SymPy, using the same pinned
package list as the existing Docker environment.

```bash
# Prints the image/packages/resources plan. No cloud client is created.
rl-zero daytona-snapshot

# Explicitly performs a remote image build and creates a billable snapshot.
rl-zero daytona-snapshot --execute
```

Copy the returned `snapshot_id` into `expected_snapshot_id` in the config. The
verifier resolves the snapshot once, records its ID and image, and creates every
sandbox from that ID. A changed name-to-ID mapping is then rejected at startup.
For an existing suitable snapshot, set `snapshot` to its name or ID and skip the
build command. It must provide `python3`, permit a root supervisor process, and
have no GPUs. Resource settings in this config apply to snapshot creation;
existing snapshot resources are read from Daytona metadata.

## Start the reward service

```bash
rl-zero verifier --backend daytona --trace-path runs/my-experiment.verifier.jsonl
```

Daytona is now the default backend. Starting the service checks snapshot metadata
but creates no candidate sandbox. Each POST request creates a sandbox, so sending
training or grading requests incurs Daytona usage.

The existing trainer `code_api_url = "http://127.0.0.1:1234/test_program"` remains
valid. Stdin/stdout requests use `/test_program_stdio`; `/health` exposes the
backend identity. Existing training and grading commands work through this gateway.

For a harmless live check **when you choose to provision one sandbox**, send:

```bash
curl -X POST http://127.0.0.1:1234/test_program \
  -H 'Content-Type: application/json' \
  -d '{"program":"def add(a,b): return a+b","tests":["assert add(2,3)==5"],"max_execution_time":1}'
```

## Execution and cleanup

Each candidate gets a fresh private sandbox with outbound network access blocked.
The root supervisor loads and deletes its uploaded test payload, closes stdin,
then runs each candidate child as UID/GID 65534 with no supplementary groups and
a minimal environment. It retains the existing per-test time budget, fractional
training reward and all-tests-pass evaluation semantics.

Deletion is attempted after success, execution errors, malformed results, and
creation timeouts that expose a sandbox ID. Deletion errors invalidate the
request and include the sandbox ID for manual cleanup. A 30-minute wall-clock TTL
provides a fallback if the gateway dies or a creation request times out before
an ID is available. Snapshot storage is separate and persists until you delete it.

The default `workers = 1` bounds concurrent sandbox usage. Increase it deliberately
for rollout throughput and account limits. A sandbox per candidate favors clean
isolation; service startup/build latency can dominate small tasks. The existing
900-second verifier client timeout also includes waiting for the gateway queue.
Raise `RL_ZERO_VERIFIER_TIMEOUT` for a workload that needs a longer queue budget.

Trace records include sandbox ID, snapshot ID, runner hash, SDK version, resource
identity, tests/results and latency. Evaluation records the Daytona identity;
comparisons with different environments or Docker graders are rejected.

The optional local backend remains available as `rl-zero verifier --backend docker`.
Its shared runner now lives in `src/rl_zero/sandbox_runner.py`.

## Verification status

The SDK models and local HTTP integration were checked offline, including timeout
cleanup and payload isolation. **No live Daytona API calls, snapshot builds,
sandbox execution, or GPU work were performed during this change.** A live smoke
check with your credentials is still needed before training.

References: [Daytona image/snapshot creation](https://www.daytona.io/docs/en/python-sdk/sync/daytona/),
[snapshot service](https://www.daytona.io/docs/en/python-sdk/sync/snapshot/),
[remote process execution](https://www.daytona.io/docs/en/python-sdk/sync/process/).
