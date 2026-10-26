"""Code RLVR reward: mean per-test success, with an optional minimum threshold.

This module never executes generated code. Execution belongs to a sandbox.
Evaluation task success requires ALL tests, even when training rewards are fractional.
"""

import base64
import io
import json
import math
import pickle
import re
import zlib


class _PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Compressed tests may contain only primitive values, never Python globals")


def _decode_compressed(value):
    compressed = base64.b64decode(value, validate=True)
    decoder = zlib.decompressobj()
    decoded = decoder.decompress(compressed, 64 * 1024 * 1024)
    if not decoder.eof or decoder.unused_data:
        raise ValueError("Compressed tests exceed 64 MiB or contain trailing data")
    try:
        return _PrimitiveUnpickler(io.BytesIO(decoded)).load()
    except (pickle.UnpicklingError, EOFError) as error:
        raise ValueError("Invalid compressed test payload") from error


def decode_tests(value):
    """Decode nested JSON or primitive-only compressed Dolci tests.

    Assertion tests are strings; stdin/stdout tests contain input/output strings.
    Compressed decoding cannot resolve or instantiate Python globals.
    """
    for _ in range(8):
        if isinstance(value, str):
            value = json.loads(value) if value.lstrip().startswith("[") else _decode_compressed(value)
        elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
            candidate = value[0].strip()
            if candidate.startswith("["):
                value = json.loads(candidate)
            elif re.fullmatch(r"[A-Za-z0-9+/]{16,}={0,2}", candidate):
                value = _decode_compressed(candidate)
            else:
                break
        else:
            break
    if not isinstance(value, list) or not value:
        raise ValueError("Expected a nonempty list of tests")
    if all(isinstance(t, dict) for t in value):
        value = [dict(t) for t in value]
        for test in value:
            for key in ("input", "output"):
                lines = test.get(key)
                if isinstance(lines, list) and all(isinstance(line, str) for line in lines):
                    test[key] = "\n".join(lines)
                elif key == "output" and isinstance(lines, (int, float)) and math.isfinite(lines):
                    test[key] = str(lines)
    assertions = all(isinstance(t, str) and t.strip() for t in value)
    stdio = all(
        isinstance(t, dict) and isinstance(t.get("input"), str) and isinstance(t.get("output"), str)
        for t in value
    )
    if not assertions and not stdio:
        raise ValueError("Expected assertion strings or stdin/stdout test dictionaries")
    return value


def extract_code(completion):
    """Match the pinned Open Instruct CodeVerifier's last-fenced-block behavior."""
    blocks = re.findall(r"```(?:python)?(.*?)```", completion, re.DOTALL)
    return blocks[-1].strip() if blocks else completion


def execution_reward(results, threshold=0.0):
    if not results or any(value not in (0, 1) for value in results):
        raise ValueError("A reward requires nonempty binary test results")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Reward threshold must be in [0, 1]")
    rate = sum(results) / len(results)
    return rate if rate >= threshold else 0.0
