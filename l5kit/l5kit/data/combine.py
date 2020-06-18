import numpy as np

from .zarr_dataset import SCENE_DTYPE


def get_combined_scenes(scenes: np.ndarray) -> np.ndarray:
    """Takes as input an np.ndarray or zarr array with scenes, and combines scenes that follow up
eachother perfectly (i.e. from a single recording by the same host). Returns an np.ndarray
of combined scenes.

    Arguments:
        scenes (np.ndarray): scenes

    Returns:
        np.ndarray -- combined scenes where followup scenes have been merged.
    """

    if len(scenes) == 0:
        return scenes

    combined_scenes = []
    start_scene = scenes[0]
    end_scene = scenes[0]

    for s in scenes[1:]:
        time_diff = s["start_time"] - end_scene["end_time"]

        if time_diff != 0 or s["host"] != start_scene["host"]:
            interval = np.array([start_scene["frame_index_interval"][0], end_scene["frame_index_interval"][1]])

            combined_scene = (interval, start_scene["host"], start_scene["start_time"], end_scene["end_time"])
            combined_scenes.append(combined_scene)
            start_scene = s

        end_scene = s

    # append the last remaining scene
    combined_scenes.append(
        (
            np.array([start_scene["frame_index_interval"][0], end_scene["frame_index_interval"][1]]),
            start_scene["host"],
            start_scene["start_time"],
            end_scene["end_time"],
        )
    )

    combined_scenes = np.array(combined_scenes, dtype=SCENE_DTYPE)
    return combined_scenes
