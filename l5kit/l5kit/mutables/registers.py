from typing import Dict

import catalogue


RUNTIME_PARAMS_REGISTER: str = "runtime_params"


registry: Dict[str, catalogue.Registry] = {
    RUNTIME_PARAMS_REGISTER: catalogue.create("mutables", RUNTIME_PARAMS_REGISTER)
}
