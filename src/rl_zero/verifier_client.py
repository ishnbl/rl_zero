"""A verifier outage is an experiment failure, never an incorrect model answer."""

import json
import urllib.request

from rl_zero.rewards import decode_tests, execution_reward


def request_execution(url, program, tests, per_test_seconds=1.0, timeout=900):
    tests = decode_tests(tests)
    if isinstance(tests[0], dict) and not url.endswith("_stdio"):
        url += "_stdio"
    request = urllib.request.Request(
        url,
        json.dumps({"program": program, "tests": tests, "max_execution_time": per_test_seconds}).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if len(result["results"]) != len(tests) or len(result["runtimes"]) != len(tests):
        raise ValueError("Verifier returned an incomplete test vector")
    execution_reward(result["results"])
    return result
