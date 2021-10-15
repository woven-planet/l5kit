import copy
import importlib
import os
from typing import Any, Dict, List, Optional, Protocol, Tuple

import l5kit.mutables.registers as registers


_RUNTIME_PARAMS: Optional[Dict[str, Any]] = None


class Prioritizes(Protocol):
    """Base interface to prioritize which loader to choose for each type of registry"""

    def get_prioritized_item(self, func_names: List[str]) -> str:
        """Gets a list of avaiable registered items and returns the most prioritized one"""
        raise NotImplementedError


class VersionPrioritizer(Prioritizes):
    """Prioritizes higher version numbers over lower ones, eg. 'v1' > 'v0'"""
    def __init__(self) -> None:
        super().__init__()
    
    def get_prioritized_item(self, versions: List[str]) -> str:
        """Accepts versions in either format of 'v<INT>' as 'v0' or '<INT>' as '0'"""
        highest_version_idx: int = sorted(
            range(len(versions)),
            key=lambda idx: int(versions[idx][1:]) if versions[idx].lower().startswith('v') else int(versions[idx]),
            reverse=True)[0]
        return versions[highest_version_idx]


def _obtain_loader(registry_name: str, prioritizer: Prioritizes = VersionPrioritizer()) -> Tuple[Any, str]:
    """
    Obtains the most prioritized loader according to the given prioritizer
    
    :returns: Tuple of the loader function, and its corresponing key in the given registry
    """
    module_path: str = os.getenv(f'PROTEAN_MODULE_{registry_name.upper()}', f'l5kit.mutables.registry.{registry_name}')
    spec = importlib.util.find_spec(module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    available_fcns = registers.registry[registry_name].get_all()
    prioritized_item: str = prioritizer.get_prioritized_item(list(available_fcns.keys()))
    print(registers.registry[registry_name].find(prioritized_item))
    return available_fcns[prioritized_item], prioritized_item


def get_runtime_params() -> Dict[str, Any]:
    """Obtains runtime param"""
    global _RUNTIME_PARAMS
    if _RUNTIME_PARAMS is None:
        loader_fcn, _ = _obtain_loader(registers.RUNTIME_PARAMS_REGISTER)
        _RUNTIME_PARAMS = loader_fcn()
    return copy.deepcopy(_RUNTIME_PARAMS)

