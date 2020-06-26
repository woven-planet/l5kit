import pytest

from l5kit.data import ChunkedStateDataset, get_frames_agents


@pytest.fixture(scope="module")
def zarr_dataset() -> ChunkedStateDataset:
    zarr_dataset = ChunkedStateDataset(path="./l5kit/tests/data/single_scene.zarr")
    zarr_dataset.open()
    return zarr_dataset


@pytest.mark.parametrize("frame_bound", [0, 10, 50, 100])
def test_get_frames_agents_shape(frame_bound: int, zarr_dataset: ChunkedStateDataset) -> None:
    agents = get_frames_agents(zarr_dataset.frames[0:frame_bound], zarr_dataset.agents)
    assert len(agents) == frame_bound


@pytest.mark.parametrize("frame_bound", [pytest.param(0, marks=pytest.mark.xfail), 10, 50, 100])
def test_get_frames_agents_ret(frame_bound: int, zarr_dataset: ChunkedStateDataset) -> None:
    agents = get_frames_agents(zarr_dataset.frames[0:frame_bound], zarr_dataset.agents)
    assert sum([len(agents_fr) for agents_fr in agents]) > 0
