import pytest

from l5kit.data import ChunkedDataset, get_frames_slice_from_scenes, LocalDataManager
from l5kit.sampling.agent_sampling_vectorized import generate_agent_sample_vectorized
from l5kit.vectorization.vectorizer import Vectorizer
from l5kit.vectorization.vectorizer_builder import build_vectorizer


def test_vectorizer_builder(dmg: LocalDataManager, cfg: dict) -> None:
    # default call should work
    vectorizer = build_vectorizer(cfg, dmg)
    assert isinstance(vectorizer, Vectorizer)

    # vectorizer requires meta to build the map
    cfg["raster_params"]["dataset_meta_key"] = "invalid_path"
    with pytest.raises(FileNotFoundError):
        build_vectorizer(cfg, dmg)


@pytest.mark.parametrize("history_num_frames_ego", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("history_num_frames_agents", [0, 1, 2, 3, 4])
def test_vectorizer_output_shape(zarr_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict,
                                 history_num_frames_ego: int, history_num_frames_agents: int) -> None:
    cfg["model_params"]["history_num_frames_ego"] = history_num_frames_ego
    cfg["model_params"]["history_num_frames_agents"] = history_num_frames_agents
    max_history_num_frames = max(history_num_frames_ego, history_num_frames_agents)
    num_agents = cfg["data_generation_params"]["other_agents_num"]

    frames = zarr_dataset.frames[get_frames_slice_from_scenes(zarr_dataset.scenes[0])]
    data = generate_agent_sample_vectorized(0, frames, zarr_dataset.agents, zarr_dataset.tl_faces, None,
                                            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
                                            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
                                            future_num_frames=cfg["model_params"]["future_num_frames"],
                                            step_time=cfg["model_params"]["step_time"],
                                            filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"],
                                            vectorizer=build_vectorizer(cfg, dmg))

    assert data["history_positions"].shape == (max_history_num_frames + 1, 2)
    assert data["history_yaws"].shape == (max_history_num_frames + 1, 1)
    assert data["history_extents"].shape == (max_history_num_frames + 1, 2)
    assert data["history_availabilities"].shape == (max_history_num_frames + 1,)

    assert data["all_other_agents_history_positions"].shape == (num_agents, max_history_num_frames + 1, 2)
    assert data["all_other_agents_history_yaws"].shape == (num_agents, max_history_num_frames + 1, 1)
    assert data["all_other_agents_history_extents"].shape == (num_agents, max_history_num_frames + 1, 2)
    assert data["all_other_agents_history_availability"].shape == (num_agents, max_history_num_frames + 1,)

    assert data["target_positions"].shape == (cfg["model_params"]["future_num_frames"], 2)
    assert data["target_yaws"].shape == (cfg["model_params"]["future_num_frames"], 1)
    assert data["target_extents"].shape == (cfg["model_params"]["future_num_frames"], 2)
    assert data["target_availabilities"].shape == (cfg["model_params"]["future_num_frames"],)

    assert data["all_other_agents_future_positions"].shape == (num_agents, cfg["model_params"]["future_num_frames"], 2)
    assert data["all_other_agents_future_yaws"].shape == (num_agents, cfg["model_params"]["future_num_frames"], 1)
    assert data["all_other_agents_future_extents"].shape == (num_agents, cfg["model_params"]["future_num_frames"], 2)
    assert data["all_other_agents_future_availability"].shape == (num_agents, cfg["model_params"]["future_num_frames"],)
    assert data["all_other_agents_types"].shape == (num_agents,)

    assert data["agent_trajectory_polyline"].shape == (max_history_num_frames + 1, 3)
    assert data["agent_polyline_availability"].shape == (max_history_num_frames + 1,)
    assert data["other_agents_polyline"].shape == (num_agents, max_history_num_frames + 1, 3)
    assert data["other_agents_polyline_availability"].shape == (num_agents, max_history_num_frames + 1,)
