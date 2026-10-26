"""Runtime-only observers. They do not change the optimizer, loss, or rewards."""

import json
import math
import numbers
import os
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()


def append_metrics(run_dir, data, step=None, stream="metrics"):
    scalars = {}
    for key, value in data.items():
        if isinstance(value, numbers.Real):
            scalars[key] = float(value) if math.isfinite(float(value)) else str(value)
    if not scalars:
        return
    record = {"time": time.time(), "step": step, "metrics": scalars}
    with _LOCK, (Path(run_dir) / f"{stream}.jsonl").open("a") as output:
        output.write(json.dumps(record, allow_nan=False) + "\n")


def install_tracking_observer(run_dir):
    """Observe Run.log (also used by Accelerate) and SFT's DeepSpeed step norm."""
    import wandb

    original_log = wandb.sdk.wandb_run.Run.log

    def observed_log(run, data, step=None, *args, **kwargs):
        append_metrics(run_dir, data, step=step)
        return original_log(run, data, step, *args, **kwargs)

    wandb.sdk.wandb_run.Run.log = observed_log

    # RLVR exposes optim/grad_norm natively from the remote DeepSpeed workers.
    # This additional observer covers SFT, where Accelerate owns DeepSpeed.step.
    import deepspeed

    original_step = deepspeed.DeepSpeedEngine.step

    def observed_step(engine, *args, **kwargs):
        result = original_step(engine, *args, **kwargs)
        if int(os.environ.get("RANK", "0")) == 0:
            norm = engine.get_global_grad_norm()
            if norm is not None:
                append_metrics(
                    run_dir, {"optim/grad_norm": float(norm)}, step=engine.global_steps, stream="gradients"
                )
        return result

    deepspeed.DeepSpeedEngine.step = observed_step
