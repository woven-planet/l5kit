import unittest

import numpy as np
import numpy.testing

from l5kit.data import zarr_dataset


class ZarrDatasetTest(unittest.TestCase):
    def test_load_dataset(self) -> None:
        data = zarr_dataset.ChunkedStateDataset(path="./l5kit/tests/data/single_scene.zarr")
        data.open()

        self.assertEqual(len(data.frames), 775)
        self.assertEqual(len(data.agents), 67954)
        self.assertEqual(len(data.scenes), 1)

        frame = data.frames[0]

        self.assertEqual(frame["timestamp"], 1266597039003039366)
        numpy.testing.assert_allclose(frame["agent_index_interval"], (0, 88))
        numpy.testing.assert_allclose(
            frame["ego_translation"], np.array([542.73755, -2405.4773, 288.671], dtype=np.float32)
        )
        numpy.testing.assert_allclose(
            frame["ego_rotation"],
            np.array(
                [[-0.432687, -0.901447, 0.013263], [0.901538, -0.432583, 0.010026], [-0.003301, 0.016295, 0.999862]],
                dtype=np.float32,
            ),
            rtol=1e-04,
        )
