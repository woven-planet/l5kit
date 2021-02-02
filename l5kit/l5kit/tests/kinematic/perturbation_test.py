import numpy as np
import pytest

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.kinematic import AckermanPerturbation
from l5kit.random import ReplayRandomGenerator
from l5kit.rasterization import build_rasterizer


@pytest.mark.parametrize("perturb_prob", [1.0, pytest.param(0.0, marks=pytest.mark.xfail)])
def test_perturbation_is_applied(
        perturb_prob: float, dmg: LocalDataManager, cfg: dict, zarr_dataset: ChunkedDataset
) -> None:
    rasterizer = build_rasterizer(cfg, dmg)

    dataset = EgoDataset(cfg, zarr_dataset, rasterizer, None)  # no perturb
    data_no_perturb = dataset[0]

    # note we cannot change the object we already have as a partial is built at init time
    perturb = AckermanPerturbation(ReplayRandomGenerator(np.asarray([[4.0, 1.0, 0.33]])), perturb_prob=perturb_prob)
    dataset = EgoDataset(cfg, zarr_dataset, rasterizer, perturb)  # perturb
    data_perturb = dataset[0]

    assert np.linalg.norm(data_no_perturb["target_positions"] - data_perturb["target_positions"]) > 0
    assert np.linalg.norm(data_no_perturb["target_yaws"] - data_perturb["target_yaws"]) > 0
