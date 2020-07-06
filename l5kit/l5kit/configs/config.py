import yaml


def load_config_data(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def save_config_data(data: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)
