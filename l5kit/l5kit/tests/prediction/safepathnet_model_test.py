import pytest
import torch
from torch import nn

from l5kit.configs import load_config_data
from l5kit.prediction.vectorized.safepathnet_model import SafePathNetModel
from l5kit.tests.planning.common_test import mock_vectorizer_data


batch_size = 15


@pytest.fixture(scope="session")
def cfg() -> dict:
    """Get a config file from artefacts.
    Note: the scope of this fixture is "function"-> one per test function

    :return: the config python dict
    """
    return load_config_data("./l5kit/tests/artefacts/config_safepathnet.yaml")


@pytest.fixture(scope="session")
def data_batch(cfg: dict) -> dict:
    """Mocks the output of the vectorizer to create a data batch.
    """
    num_steps = cfg["model_params"]["future_num_frames"]
    num_history = max(cfg["model_params"]["history_num_frames_ego"], cfg["model_params"]["history_num_frames_agents"])
    num_agents = cfg["data_generation_params"]["other_agents_num"]
    num_lanes = cfg["data_generation_params"]["lane_params"]["max_num_lanes"]
    num_crosswalks = cfg["data_generation_params"]["lane_params"]["max_num_crosswalks"]
    num_points_per_element = max(cfg["data_generation_params"]["lane_params"]["max_points_per_lane"],
                                 cfg["data_generation_params"]["lane_params"]["max_points_per_crosswalk"])

    TYPE_MAX = 99  # generate dummy types up to this index

    return mock_vectorizer_data(batch_size, num_steps, num_history, num_agents, num_lanes, num_crosswalks,
                                num_points_per_element, TYPE_MAX)


@pytest.fixture(scope="session")
def model(cfg: dict) -> SafePathNetModel:
    weights_scaling = [1.0, 1.0, 1.0]
    _num_predicted_frames = cfg["model_params"]["future_num_frames"]

    model = SafePathNetModel(
        history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
        history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
        num_timesteps=_num_predicted_frames,
        weights_scaling=weights_scaling,
        criterion=nn.L1Loss(reduction="none"),
        disable_other_agents=cfg["model_params"]["disable_other_agents"],
        disable_map=cfg["model_params"]["disable_map"],
        disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
        agent_num_trajectories=cfg["model_params"]["agent_num_trajectories"],
        max_num_agents=cfg["data_generation_params"]["other_agents_num"],
        cost_prob_coeff=cfg["model_params"]["cost_prob_coeff"],
    )

    return model


def test_model_train(model: SafePathNetModel, data_batch: dict) -> None:
    res = model(data_batch)
    assert "loss" in res
    assert res['loss'] > 0


def test_model_eval(model: SafePathNetModel, data_batch: dict, cfg: dict) -> None:
    model = model.eval()
    res = model(data_batch)
    assert 'agent_positions' in res
    assert 'agent_yaws' in res
    assert 'all_agent_positions' in res
    assert 'all_agent_yaws' in res
    assert 'agent_traj_logits' in res
    num_agents = cfg["data_generation_params"]["other_agents_num"]
    agent_num_trajectories = cfg["model_params"]["agent_num_trajectories"]
    future_num_frames = cfg["model_params"]["future_num_frames"]
    assert res['agent_positions'].shape == torch.Size([
        batch_size, num_agents, future_num_frames, 2])
    assert res['agent_yaws'].shape == torch.Size([
        batch_size, num_agents, future_num_frames, 1])
    assert res['all_agent_positions'].shape == torch.Size([
        batch_size, num_agents, agent_num_trajectories, future_num_frames, 2])
    assert res['all_agent_yaws'].shape == torch.Size([
        batch_size, num_agents, agent_num_trajectories, future_num_frames, 1])
    assert res['agent_traj_logits'].shape == torch.Size([
        batch_size, num_agents, agent_num_trajectories])
