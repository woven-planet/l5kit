from typing import Any, Dict

import os
import tempfile
import time

import l5kit.mutables.registers as registers


runtime_registry = registers.registry[registers.RUNTIME_PARAMS_REGISTER]


@runtime_registry.register("v0")
def load_runtime_params() -> Dict[str, Any]:
    """Returns a dict of default runtime params"""
    user: str = os.environ.get("USER", "DEFAULT_USER")
    job_name: str =  f"{user}-{int(time.time())}"
    log_dir: str = tempfile.mkdtemp(prefix=f"{job_name}_")
    return {
        "username": user,
        "infra_job_name": job_name,
        "experiment_name": job_name,
        "log_dir": log_dir,
        "checkpoint_dir": os.path.join(log_dir, 'checkpoints'),
    }

