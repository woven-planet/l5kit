import logging
from typing import Any, Dict


def init_logger(**kwargs: Any) -> None:
    # Read level directly from kwargs, and if not available from runtime_params.
    logging.basicConfig(level=kwargs.get("level", kwargs.get("runtime_params", {}).get("level", "INFO")))


def get_logger(name: str) -> 'L5KitLogAdapter':
    return L5KitLogAdapter(logging.getLogger(name))


class L5KitLogAdapter(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger):
        super().__init__(logger, {})
        self._custom_loggers: Dict[str, Any] = {}

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        super().log(level, msg, *args, **kwargs)

    def log_custom(self, type: str, **kwargs: Any) -> None:
        if type not in self._custom_loggers:
            return
        self._custom_loggers[type](**kwargs)
