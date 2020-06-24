from l5kit.data import ChunkedStateDataset
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

        zarr_write = ChunkedStateDataset(str(dst_path / zarr_path.name))
        zarr_write.initialize()  # create writable 0len arrays

        idx_write_frame = 0  # the real frame index (account for scenes drop)
        idx_write_agent = 0

        for idx_scene, scene in enumerate(tqdm(zarr_read.scenes, f"writing zarr {zarr_path.name}")):
            if idx_scene in black_list:
                # do not write but update write indexes with how many items are being lost
                scene = zarr_read.scenes[idx_scene]
                idx_write_frame += (scene["frame_index_interval"][1] - scene["frame_index_interval"][0])

                frame_start = zarr_read.frames[scene["frame_index_interval"][0]]
                frame_end = zarr_read.frames[scene["frame_index_interval"][1] - 1]
                idx_write_agent += (frame_end["agent_index_interval"][1] - frame_start["agent_index_interval"][0])

            else:
                # update indexes and append to dataset
                scene = np.asarray(scene)
                frames = np.asarray(zarr_read.frames[scene["frame_index_interval"][0]: scene["frame_index_interval"][1]])
                agents = zarr_read.agents[frames[0]["agent_index_interval"][0]: frames[-1]["agent_index_interval"][1]]

                scene["frame_index_interval"] -= idx_write_frame
                zarr_write.scenes.append(scene[None, ...])

                frames["agent_index_interval"] -= idx_write_agent
                zarr_write.frames.append(frames)

                zarr_write.agents.append(agents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarrs", type=Path, required=True, nargs="+")
    parser.add_argument("--blacklists", type=Path, required=True, nargs="+")
    parser.add_argument("--dst_path", type=Path, required=True)
    args = parser.parse_args()
    main(**vars(args))
