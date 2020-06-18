import os
import unittest
from tempfile import TemporaryDirectory

from l5kit.data import LocalDataManager


class TestDataManager(unittest.TestCase):
    def test_require_existing_file(self) -> None:
        with TemporaryDirectory() as directory:
            p = os.path.join(directory, "my_file.txt")
            open(p, "w").write("hello")

            dm_local = LocalDataManager(directory)
            self.assertEqual(dm_local.require("my_file.txt"), p)

    def test_require_non_existing_file(self) -> None:
        dm_local = LocalDataManager("")  # cur path
        with self.assertRaises(FileNotFoundError):
            dm_local.require("some-file-that-doesnt-exist.txt")
