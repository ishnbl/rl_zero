"""Optional CPU integration check for Ray hook transport and verifier-error propagation.

Install the trainer's pinned Ray version in the lightweight environment to run.
This does not load Open Instruct, torch, model weights, or a GPU.
"""

import os

import ray

from rl_zero.runtime_hooks import install_ray_worker_hook
from rl_zero.verifier_client import request_execution


def probe_setup():
    os.environ["RL_ZERO_CPU_HOOK_PROBE"] = "installed"


def main():
    install_ray_worker_hook(probe_setup)
    ray.init(num_cpus=1, num_gpus=0, include_dashboard=False)
    try:

        @ray.remote
        def probe():
            assert os.environ["RL_ZERO_CPU_HOOK_PROBE"] == "installed"
            assert request_execution is not None
            return "worker hook installed"

        @ray.remote
        def failed_verifier():
            # A nonexistent local endpoint must propagate a failure through Ray.
            request_execution("http://127.0.0.1:1/test_program", "", ["assert True"], timeout=1)

        print(ray.get(probe.remote()))
        try:
            ray.get(failed_verifier.remote())
        except ray.exceptions.RayTaskError:
            print("verifier infrastructure exception propagated through Ray")
        else:
            raise AssertionError("Verifier failure was swallowed")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
