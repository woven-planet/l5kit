import pytest
import torch
from torch import nn

from l5kit.configs import load_config_data
from l5kit.planning.vectorized.closed_loop_model import VectorizedUnrollModel
from l5kit.tests.planning.common_test import mock_vectorizer_data


batch_size = 15  # TODO (@lberg): don't use global


# TODO (@lberg): remove probably, we can use just one
@pytest.fixture(scope="session")
def cfg() -> dict:
    """Get a config file from artefacts.
    Note: the scope of this fixture is "function"-> one per test function

    :return: the config python dict
    """
    return load_config_data("./l5kit/tests/artefacts/config_vectorized.yaml")


@pytest.fixture(scope="session")
def data_batch(cfg: dict) -> dict:
    """Mocks the output of the vectorizer to create a data batch.
    """
    num_steps = cfg["model_params"]["future_num_frames"]
    num_history = 3
    num_agents = 30
    num_lanes = 60
    num_crosswalks = 20
    num_points_per_element = 20

    TYPE_MAX = 99  # TODO

    return mock_vectorizer_data(batch_size, num_steps, num_history, num_agents, num_lanes, num_crosswalks,
                                num_points_per_element, TYPE_MAX)


@pytest.fixture(scope="session")
def model(cfg: dict) -> VectorizedUnrollModel:
    weights_scaling = [1.0, 1.0, 1.0]
    _num_predicted_frames = cfg["model_params"]["future_num_frames"]
    _num_predicted_params = len(weights_scaling)

    model = VectorizedUnrollModel(
        history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
        history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
        num_targets=_num_predicted_params * _num_predicted_frames,
        weights_scaling=weights_scaling,
        criterion=nn.L1Loss(reduction="none"),
        global_head_dropout=cfg["model_params"]["global_head_dropout"],
        disable_other_agents=cfg["model_params"]["disable_other_agents"],
        disable_map=cfg["model_params"]["disable_map"],
        disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
        detach_unroll=False,
        warmup_num_frames=cfg["model_params"]["warmup_num_frames"],
        discount_factor=cfg["model_params"]["discount_factor"],
    )

    return model


def test_model_train(model: VectorizedUnrollModel, data_batch: dict) -> None:
    res = model(data_batch)
    assert "loss" in res
    assert res['loss'] > 0


def test_model_eval(model: VectorizedUnrollModel, data_batch: dict, cfg: dict) -> None:
    model = model.eval()
    data_batch["future_num_frames"] = cfg["model_params"]["future_num_frames"]
    res = model(data_batch)
    assert 'positions' in res
    assert 'yaws' in res
    assert res['positions'].shape == torch.Size([batch_size, cfg["model_params"]["future_num_frames"], 2])
    assert res['yaws'].shape == torch.Size([batch_size, cfg["model_params"]["future_num_frames"], 1])
