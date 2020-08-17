from pathlib import Path
from shutil import rmtree
from typing import Iterator

import pytest

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager


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


@pytest.fixture(scope="session")
def zarr_dataset(dmg: LocalDataManager) -> ChunkedDataset:
    zarr_path = dmg.require("single_scene.zarr")
    zarr_dataset = ChunkedDataset(path=zarr_path)
    zarr_dataset.open()
    return zarr_dataset


@pytest.fixture(scope="session", autouse=True)
def clean_mask(zarr_dataset: ChunkedDataset) -> Iterator[None]:
    """
    Auto clean agents_mask for artefacts during tests tear down
    """
    agents_mask_path = Path(zarr_dataset.path) / "agents_mask"
    if agents_mask_path.exists():
        rmtree(str(agents_mask_path))
    yield None
    # remove agents mask during tear down
    agents_mask_path = Path(zarr_dataset.path) / "agents_mask"
    if agents_mask_path.exists():
        rmtree(str(agents_mask_path))
