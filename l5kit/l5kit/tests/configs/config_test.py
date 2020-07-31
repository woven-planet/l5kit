from pathlib import Path

from l5kit.configs.config import load_config_data, save_config_data


def test_load_config_data() -> None:
    cfg = load_config_data("./l5kit/tests/artefacts/config.yaml")
    assert isinstance(cfg, dict)


def test_save_config_data(tmp_path: Path, cfg: dict) -> None:
    tmp_path = tmp_path / "default.yaml"
    save_config_data(cfg, str(tmp_path))
    assert tmp_path.exists()
