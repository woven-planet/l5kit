import os
import tempfile
import time
from typing import Any, Dict


class RuntimeParams:
    _RUNTIME_PARAMS: Dict[str, Any] = {}

    @classmethod
    def get(cls) -> Dict[str, Any]:
        if not RuntimeParams._RUNTIME_PARAMS:
            RuntimeParams._RUNTIME_PARAMS = cls._default()
        return RuntimeParams._RUNTIME_PARAMS

    @classmethod
    def _default(cls) -> Dict[str, Any]:
        """Returns a dict of default runtime params"""
        user: str = os.environ.get("USER", "DEFAULT_USER")
        job_name: str = f"{user}-{int(time.time())}"
        log_dir: str = tempfile.mkdtemp(prefix=f"{job_name}_")
        return {
            "username": user,
            "infra_job_name": job_name,
            "experiment_name": job_name,
            "log_dir": log_dir,
            "checkpoint_dir": os.path.join(log_dir, 'checkpoints'),
        }
