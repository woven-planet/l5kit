from l5kit.data import ChunkedStateDataset
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import List


def main(zarrs: List[Path], rindices: List[Path], dst_path: Path):

    if not dst_path.exists():
        dst_path.mkdir(exist_ok=True, parents=True)

    # get a dict for the src zarrs
    zarrs_dict = {}
    for zarr_path in zarrs:
        zarr_read = ChunkedStateDataset(path=str(zarr_path))
        zarr_read.open()
        zarrs_dict[zarr_path.stem] = zarr_read

    for rindex in rindices:
        data = np.genfromtxt(rindex, delimiter=",")

        # ---- Get len to write so we can index instead of appending (faster!)
        len_scenes = len(data)
        len_frames, len_agents = 0, 0

        for zarr_read_name, idx_scene in tqdm(data, "getting sizes"):
            zarr_read = zarrs_dict[zarr_read_name]

            scene = zarr_read.scenes[idx_scene]
            len_frames += (scene["frame_index_interval"][1] - scene["frame_index_interval"][0])

            frame_start = zarr_read.frames[scene["frame_index_interval"][0]]
            frame_end = zarr_read.frames[scene["frame_index_interval"][1] - 1]
            len_agents += (frame_end["agent_index_interval"][1] - frame_start["agent_index_interval"][0])

        # now create the target
        zarr_write_name = f"{rindex.stem}.zarr"
        zarr_write = ChunkedStateDataset(str(dst_path / zarr_write_name))
        zarr_write.initialize(frame_num=len_frames, agent_num=len_agents, scene_num=len_scenes)  # create writable 0len arrays

        idx_write_scene = 0
        idx_write_frame = 0
        idx_write_agent = 0

        for zarr_read_name, idx_scene in enumerate(tqdm(data, f"writing zarr {zarr_write_name}")):
            zarr_read = zarrs_dict[zarr_read_name]

            # update indexes and append to dataset
            scene = np.asarray(scene)
            frames = np.asarray(zarr_read.frames[scene["frame_index_interval"][0]: scene["frame_index_interval"][1]])
            agents = zarr_read.agents[frames[0]["agent_index_interval"][0]: frames[-1]["agent_index_interval"][1]]

            scene["frame_index_interval"][0] = idx_write_frame
            scene["frame_index_interval"][1] = idx_write_frame + len(frames)
            frames["agent_index_interval"][0] = idx_write_agent
            frames["agent_index_interval"][1] = idx_write_agent + len(agents)

            zarr_write.scenes[idx_write_scene] = scene
            zarr_write.frames[idx_write_frame: idx_write_frame+len(frames)] = frames
            zarr_write.agents[idx_write_agent: idx_write_agent+len(agents)] = agents

            idx_write_scene += 1
            idx_write_frame += len(frames)
            idx_write_agent += len(agents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarrs", type=Path, required=True, nargs="+")
    parser.add_argument("--rindices", type=Path, required=True, nargs="+")
    parser.add_argument("--dst_path", type=Path, required=True)
    args = parser.parse_args()
    main(**vars(args))
