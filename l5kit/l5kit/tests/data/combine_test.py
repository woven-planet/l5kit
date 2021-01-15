import numpy as np

from l5kit.data import get_combined_scenes, SCENE_DTYPE


def test_empty_input() -> None:
    # Empty
    scenes = np.array([], dtype=SCENE_DTYPE)
    combined_scenes = get_combined_scenes(scenes)
    assert len(combined_scenes) == 0


def test_trivial_input() -> None:
    # One scene
    scenes = np.zeros(1, dtype=SCENE_DTYPE)
    scenes[0]["host"] = "some-host"
    scenes[0]["start_time"] = 0
    scenes[0]["end_time"] = 1000
    scenes[0]["frame_index_interval"] = [0, 10]

    combined_scenes = get_combined_scenes(scenes)
    assert len(combined_scenes) == 1
    np.testing.assert_array_equal(scenes, combined_scenes)


def test_followup_scenes() -> None:
    num_scenes = 10
    scenes = np.zeros(num_scenes, dtype=SCENE_DTYPE)
    for i in range(num_scenes):
        scenes[i]["host"] = "some-host"
        scenes[i]["start_time"] = i * 1000
        scenes[i]["end_time"] = (i + 1) * 1000
        scenes[i]["frame_index_interval"] = [i * 10, (i + 1) * 10]

    combined_scenes = get_combined_scenes(scenes)
    assert len(combined_scenes) == 1
    combo_scene = combined_scenes[0]
    assert combo_scene["host"] == "some-host"
    assert combo_scene["start_time"] == 0
    assert combo_scene["end_time"] == 10000
    np.testing.assert_array_equal(combo_scene["frame_index_interval"], np.array([0, 100]))

    # To follow up they must be the same host
    scenes[1]["host"] = "some-other-host"
    combined_scenes = get_combined_scenes(scenes)
    assert len(combined_scenes) == 3

    # And their timestamps must follow up exactly
    scenes[5]["start_time"] += 1
    combined_scenes = get_combined_scenes(scenes)
    assert len(combined_scenes) == 4
