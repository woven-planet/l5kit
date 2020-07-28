from pathlib import Path

import pytest

from l5kit.data import LocalDataManager


def test_require_existing_file(tmp_path: Path) -> None:
    p = tmp_path / "my_file.txt"
    open(str(p), "w").write("hello")

    dm_local = LocalDataManager(tmp_path)
    assert dm_local.require("my_file.txt") == str(p)


def test_require_non_existing_file() -> None:
    dm_local = LocalDataManager("")  # cur path
    with pytest.raises(FileNotFoundError):
        dm_local.require("some-file-that-doesnt-exist.txt")
