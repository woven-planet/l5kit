import json
import os
import tempfile

import cv2
import numpy as np
import pytest

from l5kit.configs import load_config_data
from l5kit.data import ChunkedStateDataset, LocalDataManager, get_frames_agents
from l5kit.data.proto.road_network_pb2 import Lane, MapElement, MapFragment
from l5kit.rasterization import Rasterizer, build_rasterizer
from l5kit.sampling import get_history_slice


@pytest.fixture(scope="module")
def dataset() -> ChunkedStateDataset:
    zarr_dataset = ChunkedStateDataset(path="./l5kit/tests/data/single_scene.zarr")
    zarr_dataset.open()
    return zarr_dataset


def setup_rasterizer_artifacts_and_config(temp_folder: str, cfg: dict) -> None:
    sem_json_filename = "sem.pb"
    cfg["raster_params"]["semantic_map_key"] = sem_json_filename

    # create a single lane protobuf
    # TODO add also a crosswalk
    bnd_left = Lane.Boundary(  # type: ignore
        vertex_deltas_x_cm=[1, 2, 3], vertex_deltas_y_cm=[10, 10, 10], vertex_deltas_z_cm=[0, 0, 0]
    )
    bnd_right = Lane.Boundary(  # type: ignore
        vertex_deltas_x_cm=[1, 2, 3], vertex_deltas_y_cm=[30, 30, 30], vertex_deltas_z_cm=[0, 0, 0]
    )
    lane = Lane(left_boundary=bnd_left, right_boundary=bnd_right)
    mf = MapFragment()
    mf.elements.append(MapElement(element=MapElement.Element(lane=lane)))  # type: ignore

    with open(os.path.join(temp_folder, sem_json_filename), "wb") as fb:
        fb.write(mf.SerializeToString())

    sat_im_filename = "sat_im.png"
    sat_im_metadata_filename = "sat_im.json"
    cfg["raster_params"]["satellite_map_key"] = sat_im_filename

    cv2.imwrite(os.path.join(temp_folder, sat_im_filename), np.zeros((2000, 1000, 3), dtype=np.uint8))
    # Note(gzuidhof@): I added the x and y offsets manually so we can get away with a smaller image
    # Using a 20k by 10k image is required otherwise, which makes this test take 9 seconds, now it takes ±0.3 sec
    sat_im_transform_json_content = {
        "ecef_to_image": np.array(
            [
                [-7.17416495e-01, -1.14606296e00, -1.62854453e00, -5.72869824e05 - 9000],
                [1.80065798e00, -1.08914046e00, -2.87877303e-02, 3.00171963e05 - 6500],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        ).tolist()
    }
    with open(os.path.join(temp_folder, sat_im_metadata_filename), "w") as ft:
        json.dump(sat_im_transform_json_content, ft)


def check_rasterizer(cfg: dict, rasterizer: Rasterizer, dataset: ChunkedStateDataset) -> None:
    frames = dataset.frames[:]  # Load all frames into memory
    for current_frame in [0, 50, len(frames) - 1]:
        history_num_frames = cfg["model_params"]["history_num_frames"]
        history_step_size = cfg["model_params"]["history_step_size"]
        s = get_history_slice(current_frame, history_num_frames, history_step_size, include_current_state=True)
        frames_to_rasterize = frames[s]
        agents = get_frames_agents(frames_to_rasterize, dataset.agents)

        im = rasterizer.rasterize(frames_to_rasterize, agents)
        assert len(im.shape) == 3
        assert im.shape[:2] == tuple(cfg["raster_params"]["raster_size"])
        assert im.shape[2] >= 3
        assert im.max() <= 1
        assert im.min() >= 0
        assert im.dtype == np.float32

        rgb_im = rasterizer.to_rgb(im)
        assert im.shape[:2] == rgb_im.shape[:2]
        assert rgb_im.shape[2] == 3  # RGB has three channels
        assert rgb_im.dtype == np.uint8


@pytest.mark.parametrize("map_type", ["py_semantic", "py_satellite"])
def test_rasterizer_created_from_config(map_type: str, dataset: ChunkedStateDataset) -> None:
    cfg = load_config_data("./l5kit/configs/default.yaml")
    cfg["raster_params"]["map_type"] = map_type
    with tempfile.TemporaryDirectory("", "rasterizer-test") as tmpdir:
        setup_rasterizer_artifacts_and_config(tmpdir, cfg)
        dm = LocalDataManager(tmpdir)
        rasterizer = build_rasterizer(cfg, dm)
        check_rasterizer(cfg, rasterizer, dataset)
