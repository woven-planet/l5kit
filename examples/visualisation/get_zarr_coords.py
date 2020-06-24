import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from l5kit.data import ChunkedStateDataset, LocalDataManager
from l5kit.dataset import EgoDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from typing import List


def main(zarrs: List[Path], dst_path):
    cfg = load_config_data("./visualisation_config.yaml")

    if not dst_path.exists():
        dst_path.mkdir(exist_ok=True, parents=True)

    for zarr_path in zarrs:
        zarr_dataset = ChunkedStateDataset(str(zarr_path))
        zarr_dataset.open()
        dm = LocalDataManager()
        rast = build_rasterizer(cfg, dm)
        dataset = EgoDataset(cfg, zarr_dataset, rast)
        print(dataset)

        with open(dst_path / f"coords_{zarr_path.stem}.txt", "wt") as fp:
            for idx_scene, scene in enumerate(tqdm(zarr_dataset.scenes)):
                translations = np.asarray(zarr_dataset.frames[scene["frame_index_interval"][0]:
                                                              scene["frame_index_interval"][1]]["ego_translation"][:2])

                timestamps = np.asarray(zarr_dataset.frames[scene["frame_index_interval"][0]:
                                                              scene["frame_index_interval"][1]]["timestamp"])

                for coords, timestamp in zip(translations, timestamps):
                    fp.write(f"{coords[0]},{coords[1]},{scene['host']},{timestamp},{idx_scene}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--zarrs", required=True, type=Path, nargs="+", help="zarr path(s)")
    parser.add_argument("--dst_path", required=True, type=Path, help="dest path")

    args = parser.parse_args()
    main(**vars(args))
