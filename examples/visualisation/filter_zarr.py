from l5kit.data import ChunkedStateDataset
from l5kit.data.zarr_dataset import SCENE_ARRAY_KEY, FRAME_ARRAY_KEY, AGENT_ARRAY_KEY
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import List


def main(zarrs: List[Path], blacklists: List[Path], dst_path: Path):
    assert len(zarrs) == len(blacklists)

    if not dst_path.exists():
        dst_path.mkdir(exist_ok=True, parents=True)

    for zarr_path, bl_path in zip(zarrs, blacklists):
        print(f"using zarr:{zarr_path.name} and blacklist:{bl_path.name}")
        zarr_read = ChunkedStateDataset(path=str(zarr_path))
        zarr_read.open()

        black_list = np.genfromtxt(bl_path, dtype=int, delimiter=",")

        # ---- Get Len to write so we can index instead of appending (faster!)
        len_scenes = len(zarr_read.scenes) - len(black_list)
        len_frames = len(zarr_read.frames)
        len_agents = len(zarr_read.agents)

        for idx_scene in tqdm(black_list, "getting sizes"):
            scene = zarr_read.scenes[idx_scene]
            len_frames -= (scene["frame_index_interval"][1] - scene["frame_index_interval"][0])

            frame_start = zarr_read.frames[scene["frame_index_interval"][0]]
            frame_end = zarr_read.frames[scene["frame_index_interval"][1] - 1]
            len_agents -= (frame_end["agent_index_interval"][1] - frame_start["agent_index_interval"][0])

        zarr_write = ChunkedStateDataset(str(dst_path / zarr_path.name))
        zarr_write.initialize(frame_num=len_frames, agent_num=len_agents, scene_num=len_scenes)  # create writable 0len arrays

        idx_bound_frame = 0  # these account for scenes drop
        idx_bound_agent = 0

        idx_write_scene = 0
        idx_write_frame = 0
        idx_write_agent = 0

        for idx_scene, scene in enumerate(tqdm(zarr_read.scenes, f"writing zarr {zarr_path.name}")):
            if idx_scene in black_list:
                # do not write but update write indexes with how many items are being lost
                scene = zarr_read.scenes[idx_scene]
                idx_bound_frame += (scene["frame_index_interval"][1] - scene["frame_index_interval"][0])

                frame_start = zarr_read.frames[scene["frame_index_interval"][0]]
                frame_end = zarr_read.frames[scene["frame_index_interval"][1] - 1]
                idx_bound_agent += (frame_end["agent_index_interval"][1] - frame_start["agent_index_interval"][0])

            else:
                # update indexes and append to dataset
                scene = np.asarray(scene)
                frames = np.asarray(zarr_read.frames[scene["frame_index_interval"][0]: scene["frame_index_interval"][1]])
                agents = zarr_read.agents[frames[0]["agent_index_interval"][0]: frames[-1]["agent_index_interval"][1]]

                scene["frame_index_interval"] -= idx_bound_frame
                frames["agent_index_interval"] -= idx_bound_agent

                zarr_write.scenes[idx_write_scene] = scene
                zarr_write.frames[idx_write_frame: idx_write_frame+len(frames)] = frames
                zarr_write.agents[idx_write_agent: idx_write_agent+len(agents)] = agents

                idx_write_scene += 1
                idx_write_frame += len(frames)
                idx_write_agent += len(agents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarrs", type=Path, required=True, nargs="+")
    parser.add_argument("--blacklists", type=Path, required=True, nargs="+")
    parser.add_argument("--dst_path", type=Path, required=True)
    args = parser.parse_args()
    main(**vars(args))
