from pathlib import Path
from shutil import rmtree
from typing import Iterator
from uuid import uuid4

import pytest

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.data.zarr_utils import zarr_concat
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer


@pytest.fixture(scope="session")
def dmg() -> LocalDataManager:
    """Get a data manager for the artefacts folder.
    Note: the scope of this fixture is "session"-> only one is created regardless the number of the tests

    :return: the data manager object
    """
    return LocalDataManager("./l5kit/tests/artefacts/")


@pytest.fixture(scope="function")
def cfg() -> dict:
    """Get a config file from artefacts.
    Note: the scope of this fixture is "function"-> one per test function

    :return: the config python dict
    """
    return load_config_data("./l5kit/tests/artefacts/config.yaml")


@pytest.fixture(scope="session")
def env_cfg_path() -> str:
    """Get a L5 environment config file from artefacts.
    Note: the scope of this fixture is "session"-> one per test session

    :return: the L5Kit gym-compatible environment config python dict
    """
    env_cfg_path = "./l5kit/tests/artefacts/gym_config.yaml"
    return env_cfg_path


@pytest.fixture(scope="session")
def zarr_dataset(dmg: LocalDataManager) -> ChunkedDataset:
    zarr_path = dmg.require("single_scene.zarr")
    zarr_dataset = ChunkedDataset(path=zarr_path)
    zarr_dataset.open()
    return zarr_dataset


@pytest.fixture(scope="function")
def zarr_cat_dataset(dmg: LocalDataManager, tmp_path: Path) -> ChunkedDataset:
    concat_count = 4
    zarr_input_path = dmg.require("single_scene.zarr")
    zarr_output_path = str(tmp_path / f"{uuid4()}.zarr")

    zarr_concat([zarr_input_path] * concat_count, zarr_output_path)
    zarr_cat_dataset = ChunkedDataset(zarr_output_path)
    zarr_cat_dataset.open()
    return zarr_cat_dataset


@pytest.fixture(scope="function")
def ego_cat_dataset(cfg: dict, dmg: LocalDataManager, zarr_cat_dataset: ChunkedDataset) -> EgoDataset:
    rasterizer = build_rasterizer(cfg, dmg)
    return EgoDataset(cfg, zarr_cat_dataset, rasterizer)


@pytest.fixture(scope="session", autouse=True)
def clean_mask(zarr_dataset: ChunkedDataset) -> Iterator[None]:
    """Auto clean agents_mask for artefacts during tests tear down
    """
    agents_mask_path = Path(zarr_dataset.path) / "agents_mask"
    if agents_mask_path.exists():
        rmtree(str(agents_mask_path))
    yield None
    # remove agents mask during tear down
    agents_mask_path = Path(zarr_dataset.path) / "agents_mask"
    if agents_mask_path.exists():
        rmtree(str(agents_mask_path))
