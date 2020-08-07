from pathlib import Path
from uuid import uuid4

import numpy as np

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.data.zarr_utils import zarr_concat


def test_zarr_concat(dmg: LocalDataManager, tmp_path: Path, zarr_dataset: ChunkedDataset) -> None:
    concat_count = 4
    zarr_input_path = dmg.require("single_scene.zarr")
    zarr_output_path = str(tmp_path / f"{uuid4()}.zarr")

    zarr_concat([zarr_input_path] * concat_count, zarr_output_path)
    zarr_cat_dataset = ChunkedDataset(zarr_output_path)
    zarr_cat_dataset.open()

    # check lens of arrays
    assert len(zarr_cat_dataset.scenes) == len(zarr_dataset.scenes) * concat_count
    assert len(zarr_cat_dataset.frames) == len(zarr_dataset.frames) * concat_count
    assert len(zarr_cat_dataset.agents) == len(zarr_dataset.agents) * concat_count
    assert len(zarr_cat_dataset.tl_faces) == len(zarr_dataset.tl_faces) * concat_count

    # check the first and last element concat_count times
    input_scene_a = zarr_dataset.scenes[0]
    input_scene_b = zarr_dataset.scenes[-1]
    input_frame_a = zarr_dataset.frames[0]
    input_frame_b = zarr_dataset.frames[-1]
    input_agent_a = zarr_dataset.agents[0]
    input_agent_b = zarr_dataset.agents[-1]
    input_tl_a = zarr_dataset.tl_faces[0]
    input_tl_b = zarr_dataset.tl_faces[-1]

    for idx in range(concat_count):
        output_scene_a = zarr_cat_dataset.scenes[idx * len(zarr_dataset.scenes)]
        output_scene_b = zarr_cat_dataset.scenes[(idx + 1) * len(zarr_dataset.scenes) - 1]

        # check all scene fields
        assert output_scene_a["host"] == input_scene_a["host"]
        assert output_scene_a["start_time"] == input_scene_a["start_time"]
        assert output_scene_a["end_time"] == input_scene_a["end_time"]

        displace_frame = len(zarr_dataset.frames) * idx
        assert np.all(output_scene_a["frame_index_interval"] == input_scene_a["frame_index_interval"] + displace_frame)
        assert np.all(output_scene_b["frame_index_interval"] == input_scene_b["frame_index_interval"] + displace_frame)

        # check all the frame fields
        output_frame_a = zarr_cat_dataset.frames[idx * len(zarr_dataset.frames)]
        output_frame_b = zarr_cat_dataset.frames[(idx + 1) * len(zarr_dataset.frames) - 1]

        assert np.allclose(output_frame_a["ego_rotation"], input_frame_a["ego_rotation"])
        assert np.allclose(output_frame_b["ego_rotation"], input_frame_b["ego_rotation"])
        assert np.allclose(output_frame_a["ego_translation"], input_frame_a["ego_translation"])
        assert np.allclose(output_frame_b["ego_translation"], input_frame_b["ego_translation"])
        assert output_frame_a["timestamp"] == input_frame_a["timestamp"]
        assert output_frame_b["timestamp"] == input_frame_b["timestamp"]

        displace_agent = len(zarr_dataset.agents) * idx
        assert np.all(output_frame_a["agent_index_interval"] == input_frame_a["agent_index_interval"] + displace_agent)
        assert np.all(output_frame_b["agent_index_interval"] == input_frame_b["agent_index_interval"] + displace_agent)

        displace_tl = len(zarr_dataset.tl_faces) * idx
        assert np.all(
            output_frame_a["traffic_light_faces_index_interval"]
            == input_frame_a["traffic_light_faces_index_interval"] + displace_tl
        )
        assert np.all(
            output_frame_b["traffic_light_faces_index_interval"]
            == input_frame_b["traffic_light_faces_index_interval"] + displace_tl
        )

        # check agents
        output_agent_a = zarr_cat_dataset.agents[idx * len(zarr_dataset.agents)]
        output_agent_b = zarr_cat_dataset.agents[(idx + 1) * len(zarr_dataset.agents) - 1]
        assert output_agent_a == input_agent_a
        assert output_agent_b == input_agent_b

        # check tfl
        output_tl_a = zarr_cat_dataset.tl_faces[idx * len(zarr_dataset.tl_faces)]
        output_tl_b = zarr_cat_dataset.tl_faces[(idx + 1) * len(zarr_dataset.tl_faces) - 1]
        assert output_tl_a == input_tl_a
        assert output_tl_b == input_tl_b


def test_zarr_split() -> None:
    # TODO write test
    pass
