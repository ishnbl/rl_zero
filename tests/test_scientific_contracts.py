"""Small CPU checks for errors that could invalidate an experiment."""

import base64
import pickle
import tempfile
import threading
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError

from rl_zero.config import resolve_config
from rl_zero.data import normalize_code_row, partition_code
from rl_zero.launch import build_plan
from rl_zero.metrics import paired_changes, pass_at_k
from rl_zero.report import compare_results
from rl_zero.rewards import decode_tests, execution_reward
from rl_zero.upstream import template_text, validate_flags
from rl_zero.verifier_client import request_execution


def row(identifier, prompt="Implement add(a, b)."):
    return {
        "id": identifier,
        "prompt": prompt,
        "solution": "def add(a, b): return a + b",
        "ground_truth": ['["assert add(1, 2) == 3"]'],
    }


class ScientificContracts(unittest.TestCase):
    def test_verifier_outage_is_never_a_zero_reward(self):
        class Unavailable(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(503)
                self.end_headers()

            def log_message(self, *args):
                pass

        with HTTPServer(("127.0.0.1", 0), Unavailable) as server:
            thread = threading.Thread(target=server.handle_request)
            thread.start()
            try:
                with self.assertRaises(HTTPError):
                    request_execution(
                        f"http://127.0.0.1:{server.server_port}/test_program", "", ["assert True"]
                    )
            finally:
                thread.join(timeout=5)

    def test_stdio_tests_decode_without_loading_python_globals(self):
        tests = [{"input": "1\n", "output": "2\n"}]
        encoded = base64.b64encode(zlib.compress(pickle.dumps(tests))).decode()
        self.assertEqual(decode_tests([encoded]), tests)
        malicious = base64.b64encode(zlib.compress(pickle.dumps(eval))).decode()
        with self.assertRaises(ValueError):
            decode_tests([malicious])

    def test_shared_code_template_has_stable_sft_boundary(self):
        from jinja2 import Environment

        environment = Environment()
        root = Path(__file__).resolve().parents[1]
        prompt = [{"role": "user", "content": "Implement add(a,b)."}]
        original = template_text(root / "vendor/open-instruct", "olmo_thinker_code_rlzero")
        selected = template_text(root / "vendor/open-instruct", "olmo_lab_code")
        header = environment.from_string(selected).render(messages=prompt, add_generation_prompt=True)
        self.assertEqual(
            header, environment.from_string(original).render(messages=prompt, add_generation_prompt=True)
        )
        full = environment.from_string(selected).render(
            messages=prompt + [{"role": "assistant", "content": "SOLUTION"}],
            add_generation_prompt=False,
            eos_token="EOS",
        )
        self.assertEqual(full, header + "SOLUTIONEOS")

    def test_launch_disables_publication_and_pins_initialization(self):
        root = Path(__file__).resolve().parents[1]
        config = resolve_config(root, "configs/experiments/math_code_rlvr.toml")
        with tempfile.TemporaryDirectory() as directory:
            plan = build_plan(root, config, Path(directory) / "run")
        self.assertFalse(plan["trainer"]["push_to_hub"])
        self.assertFalse(plan["trainer"]["try_auto_save_to_beaker"])
        self.assertEqual(plan["trainer"]["model_revision"], config["sources"]["models"]["math"]["revision"])
        self.assertEqual(plan["environment"]["WANDB_MODE"], "offline")
        self.assertIn("--frozen", plan["command"])
        with self.assertRaises(ValueError):
            validate_flags(root / "vendor/open-instruct", {"clip_grad_norm": 1}, method="rlvr")

    def test_comparison_rejects_different_generation_protocols(self):
        run = {
            "protocol": {"temperature": 1},
            "tasks": {"a": 0.5},
            "k": 1,
            "benchmark_sha256": "same",
            "domain": "math",
            "grader": "same",
        }
        with self.assertRaises(ValueError):
            compare_results(run, {**run, "protocol": {"temperature": 0}})

    def test_nested_tests_and_fractional_reward(self):
        self.assertEqual(decode_tests(row("x")["ground_truth"]), ["assert add(1, 2) == 3"])
        self.assertEqual(execution_reward([1, 0, 1], threshold=0), 2 / 3)
        self.assertEqual(execution_reward([1, 0, 1], threshold=1), 0)
        with self.assertRaises(ValueError):
            execution_reward([])
        with self.assertRaises(ValueError):
            decode_tests(["[]"])

    def test_normalization_separates_solutions_from_rl_prompts(self):
        item = normalize_code_row(row("x"))
        self.assertEqual(item["messages"], [{"role": "user", "content": "Implement add(a, b)."}])
        self.assertEqual(item["ground_truth"], ["assert add(1, 2) == 3"])
        self.assertEqual(item["dataset"], "code")

    def test_prompt_duplicates_cannot_cross_partitions(self):
        rows = [row(str(i), f"Implement function {i}.") for i in range(100)]
        rows += [row("duplicate", "  Implement function 1.  ")]
        parts = partition_code(rows, seed=17, validation_fraction=0.2, test_fraction=0.2)
        ids = [{r["prompt_hash"] for r in items} for items in parts.values()]
        self.assertEqual(sum(map(len, parts.values())), 100)
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(
            parts, partition_code(reversed(rows), seed=17, validation_fraction=0.2, test_fraction=0.2)
        )

    def test_pass_at_k_is_task_success_estimator(self):
        self.assertAlmostEqual(pass_at_k(10, 2, 2), 17 / 45)
        self.assertEqual(pass_at_k(5, 0, 3), 0)
        self.assertEqual(pass_at_k(5, 5, 3), 1)
        with self.assertRaises(ValueError):
            pass_at_k(2, 1, 3)

    def test_forgetting_is_paired_and_signed(self):
        before = {"a": 1.0, "b": 1.0, "c": 0.0}
        after = {"a": 0.0, "b": 1.0, "c": 1.0}
        change = paired_changes(before, after, bootstrap_samples=100)
        self.assertEqual(change["delta_pp"], 0)
        self.assertEqual(change["regressed"], 1)
        self.assertEqual(change["improved"], 1)
        with self.assertRaises(ValueError):
            paired_changes(before, {"a": 0.0})


if __name__ == "__main__":
    unittest.main()
