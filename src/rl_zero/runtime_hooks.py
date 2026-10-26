"""Small, explicit runtime adaptations applied in the driver and every Ray worker.

Upstream's CodeVerifier converts HTTP errors to zero rewards. That behavior would
confound learning with verifier outages. This replacement preserves valid-response
reward semantics and propagates infrastructure errors. No upstream file is edited.
"""

import asyncio
import os
from pathlib import Path

from rl_zero.io import write_json
from rl_zero.rewards import execution_reward, extract_code
from rl_zero.templates import LOCAL_TEMPLATES
from rl_zero.verifier_client import request_execution


def install_worker_hooks():
    from open_instruct import dataset_transformation, ground_truth_utils

    dataset_transformation.CHAT_TEMPLATES.update(LOCAL_TEMPLATES)

    async def verify(self, tokenized_prediction, prediction, label, query=None, rollout_state=None):
        config = self.verifier_config
        try:
            result = await asyncio.to_thread(
                request_execution,
                config.code_api_url,
                extract_code(prediction),
                label,
                config.code_max_execution_time,
                float(os.environ.get("RL_ZERO_VERIFIER_TIMEOUT", "900")),
            )
            score = execution_reward(result["results"], config.code_pass_rate_reward_threshold)
            if config.code_apply_perf_penalty and score > 0:
                score = sum(
                    p * (config.code_max_execution_time - t) / config.code_max_execution_time
                    for p, t in zip(result["results"], result["runtimes"])
                ) / len(result["results"])
            return ground_truth_utils.VerificationResult(score=score)
        except Exception as error:
            run = Path(os.environ["RL_ZERO_RUN_DIR"])
            write_json(run / f"verifier_failure_{os.getpid()}.json", {"error": str(error)})
            raise RuntimeError("Code verifier infrastructure failed; this run is invalid") from error

    ground_truth_utils.CodeVerifier.async_call = verify
    if os.environ.get("RL_ZERO_RUN_DIR"):
        write_json(
            Path(os.environ["RL_ZERO_RUN_DIR"]) / f"worker_hooks_{os.getpid()}.json",
            {"shared_template": True, "fail_closed_code_verifier": True},
        )


def install_ray_worker_hook(worker_setup=install_worker_hooks):
    import ray

    original_init = ray.init

    def initialize(*args, **kwargs):
        runtime = dict(kwargs.get("runtime_env", {}))
        if runtime.get("worker_process_setup_hook"):
            raise ValueError("A worker setup hook is already configured; compose hooks explicitly")
        runtime["worker_process_setup_hook"] = worker_setup
        kwargs["runtime_env"] = runtime
        return original_init(*args, **kwargs)

    ray.init = initialize
