import numpy as np
import pytest

from l5kit.data import (ChunkedDataset, filter_agents_by_frames, filter_tl_faces_by_frames,
                        get_agents_slice_from_frames, get_frames_slice_from_scenes, get_tl_faces_slice_from_frames)
from l5kit.sampling.agent_sampling import get_agent_context


@pytest.mark.parametrize("state_index", [0, 10, 40])
@pytest.mark.parametrize("history_steps", [0, 5, 10])
@pytest.mark.parametrize("future_steps", [0, 5, 10])
def test_get_agent_context(
        zarr_dataset: ChunkedDataset, state_index: int, history_steps: int, future_steps: int
) -> None:
    scene = zarr_dataset.scenes[0]
    frames = zarr_dataset.frames[get_frames_slice_from_scenes(scene)]
    agents = zarr_dataset.agents[get_agents_slice_from_frames(*frames[[0, -1]])]
    tls = zarr_dataset.tl_faces[get_tl_faces_slice_from_frames(*frames[[0, -1]])]

    frames_his_f, frames_fut_f, agents_his_f, agents_fut_f, tls_his_f, tls_fut_f = get_agent_context(
        state_index, frames, agents, tls, history_steps, future_steps
    )

    # test future using timestamp
    first_idx = state_index + 1
    last_idx = state_index + 1 + future_steps

    frames_fut = frames[first_idx:last_idx]
    agents_fut = filter_agents_by_frames(frames_fut, zarr_dataset.agents)
    tls_fut = filter_tl_faces_by_frames(frames_fut, zarr_dataset.tl_faces)

    assert np.all(frames_fut_f["timestamp"] == frames_fut["timestamp"])

    assert len(agents_fut) == len(agents_fut_f)
    for idx in range(len(agents_fut)):
        assert np.all(agents_fut_f[idx] == agents_fut[idx])

    assert len(tls_fut) == len(tls_fut_f)
    for idx in range(len(tls_fut)):
        assert np.all(tls_fut_f[idx] == tls_fut[idx])

    # test past (which is reversed and include present)
    first_idx = max(state_index - history_steps, 0)
    last_idx = state_index + 1

    frames_his = frames[first_idx:last_idx]
    agents_his = filter_agents_by_frames(frames_his, zarr_dataset.agents)
    tls_his = filter_tl_faces_by_frames(frames_his, zarr_dataset.tl_faces)

    assert np.all(frames_his_f["timestamp"] == frames_his["timestamp"][::-1])

    assert len(agents_his) == len(agents_his_f)
    for idx in range(len(agents_his)):
        assert np.all(agents_his_f[idx] == agents_his[len(agents_his) - idx - 1])

    assert len(tls_his) == len(tls_his_f)
    for idx in range(len(tls_his)):
        assert np.all(tls_his_f[idx] == tls_his[len(tls_his) - idx - 1])
