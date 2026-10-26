"""CPU math/code grading in Open Instruct's environment; no model is loaded."""

import argparse
import json
from pathlib import Path

from rl_zero.evaluation import grade_predictions
from rl_zero.upstream import verify_checkout


def main():
    from open_instruct.ground_truth_utils import GSM8KVerifier, MathVerifier

    parser = argparse.ArgumentParser()
    for field in ("benchmark", "predictions", "output"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--api-url", default="http://127.0.0.1:1234/test_program")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    source = verify_checkout(args.root)
    metadata = json.loads(Path(args.predictions + ".meta.json").read_text())
    if metadata["grader_source"] != "open-instruct:" + source["commit"]:
        raise ValueError("Generation and math grader must use the same pinned Open Instruct source")
    math, gsm = MathVerifier(), GSM8KVerifier()

    def grade(text, answer, kind):
        verifier = gsm if kind == "gsm8k" else math
        return verifier([], text, answer).score

    result = grade_predictions(
        args.benchmark, args.predictions, args.output, args.k, api_url=args.api_url, math_grader=grade
    )
    print(f"pass@{args.k}: {result['score']:.4f}")


if __name__ == "__main__":
    main()
