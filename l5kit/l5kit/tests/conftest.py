import pytest

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
