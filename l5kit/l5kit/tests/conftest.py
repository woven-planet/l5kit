import pytest

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager


@pytest.fixture(scope="session")
def dmg() -> LocalDataManager:
    """
    Get a data manager for the artefacts folder.
    Note: the scope of this fixture is "session"-> only one is created regardless the number of the tests

    Returns:
        LocalDataManager: the data manager object
    """
    return LocalDataManager("./l5kit/tests/artefacts/")


@pytest.fixture(scope="function")
def cfg() -> dict:
    """
    Get a config file from artefacts
        Note: the scope of this fixture is "function"-> one per test function

    Returns:
        dict: the config python dict
    """
    return load_config_data("./l5kit/tests/artefacts/config.yaml")
