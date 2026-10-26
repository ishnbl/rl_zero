"""Executed ONLY by an explicit training launch, inside the upstream environment."""

import argparse
import importlib.metadata
import os
import runpy
import sys
from pathlib import Path

from rl_zero.instrumentation import install_tracking_observer
from rl_zero.io import write_json
from rl_zero.runtime_hooks import install_ray_worker_hook, install_worker_hooks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True)
    options, arguments = parser.parse_known_args()
    trainer = Path(options.trainer).resolve()
    sys.path.insert(0, str(trainer.parents[1]))
    run_dir = os.environ["RL_ZERO_RUN_DIR"]
    if int(os.environ.get("RANK", "0")) == 0:
        write_json(
            Path(run_dir) / "trainer_environment.json",
            {
                "python": sys.version,
                "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            },
        )
    install_tracking_observer(run_dir)
    install_worker_hooks()
    install_ray_worker_hook()
    sys.argv = [str(trainer), *arguments]
    runpy.run_path(str(trainer), run_name="__main__")


if __name__ == "__main__":
    main()
