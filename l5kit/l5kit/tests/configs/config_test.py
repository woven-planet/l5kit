from pathlib import Path

from l5kit.configs.config import load_config_data, save_config_data


def test_load_config_data() -> None:
    cfg = load_config_data("./l5kit/configs/default.yaml")
    assert isinstance(cfg, dict)


def test_save_config_data(tmp_path: Path) -> None:
    cfg = load_config_data("./l5kit/configs/default.yaml")
    tmp_path = tmp_path / "default.yaml"
    save_config_data(cfg, str(tmp_path))
    assert tmp_path.exists()
