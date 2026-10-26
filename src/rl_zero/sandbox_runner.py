"""Container entrypoint. Generated programs run only in children of this container."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path


def stdio_equal(actual, expected):
    """Pinned upstream line stripping and exact Decimal fallback, without float tolerance."""
    actual = [line.strip() for line in actual.strip().split("\n")]
    expected = [line.strip() for line in expected.strip().split("\n")]
    if len(actual) != len(expected):
        return False
    for found, wanted in zip(actual, expected):
        if found == wanted:
            continue
        try:
            if [Decimal(x) for x in found.split()] != [Decimal(x) for x in wanted.split()]:
                return False
        except InvalidOperation:
            return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload")
    parser.add_argument("--unprivileged-children", action="store_true")
    args = parser.parse_args(argv)
    credentials = {}
    if args.unprivileged_children:
        if os.geteuid() != 0:
            raise RuntimeError("The Daytona runner must start as root to drop candidate privileges")
        credentials = {"user": 65534, "group": 65534, "extra_groups": []}
    if args.payload:
        with Path(args.payload).open() as stream:
            payload = json.load(stream)
        Path(args.payload).unlink()
    else:
        payload = json.load(sys.stdin)
    # Do not leave tests accessible through an uploaded file or /proc/<runner>/fd/0.
    sys.stdin.close()
    results, runtimes = [], []
    for test in payload["tests"]:
        start = time.monotonic()
        stdio = isinstance(test, dict)
        marker = "RL_ZERO_TEST_PASSED_" + uuid.uuid4().hex
        # Requiring a post-test marker keeps exit(0) in a candidate from passing an assertion test.
        code = (
            payload["program"]
            if stdio
            else (
                "exec("
                + repr(payload["program"])
                + ")\nexec("
                + repr(test)
                + ")\nprint("
                + repr(marker)
                + ")"
            )
        )
        # A new interpreter per test prevents one test from mutating another's state.
        # Discard output on disk to prevent unlimited captured output in RAM.
        with tempfile.TemporaryFile() as output:
            try:
                process = subprocess.run(
                    [sys.executable, "-I", "-c", code],
                    stdout=output,
                    stderr=subprocess.DEVNULL,
                    input=test["input"].encode() if stdio else b"",
                    timeout=payload["max_execution_time"],
                    check=False,
                    env={
                        "PATH": "/usr/local/bin:/usr/bin:/bin",
                        "HOME": "/tmp",
                        "LANG": "C.UTF-8",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                    },
                    **credentials,
                )
                output.seek(0)
                captured = output.read(8 * 1024 * 1024).decode(errors="replace")
                valid_output = (
                    stdio_equal(captured, test["output"]) if stdio else captured.rstrip().endswith(marker)
                )
                passed = int(process.returncode == 0 and valid_output)
            except subprocess.TimeoutExpired:
                passed = 0
        results.append(passed)
        runtimes.append(time.monotonic() - start)
    print(json.dumps({"results": results, "runtimes": runtimes}))


if __name__ == "__main__":
    main()
