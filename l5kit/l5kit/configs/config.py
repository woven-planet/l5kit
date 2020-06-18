from pathlib import Path

from strictyaml import Bool, Float, Int, Map, Seq, Str, as_document, load

model_params = Map(
    {
        "model_architecture": Str(),
        "history_num_frames": Int(),
        "history_step_size": Int(),
        "history_delta_time": Float(),
        "future_num_frames": Int(),
        "future_step_size": Int(),
        "future_delta_time": Float(),
    }
)

raster_params = Map(
    {
        "raster_size": Seq(Int()),
        "pixel_size": Seq(Float()),
        "ego_center": Seq(Float()),
        "map_type": Str(),
        "satellite_map_key": Str(),
        "semantic_map_key": Str(),
        "filter_agents_threshold": Float(),
    }
)

dataset_params = Map({"key": Str(), "scene_indices": Seq(Int())})

data_loader = Map(
    {
        "datasets": Seq(dataset_params),
        "perturb_probability": Float(),
        "batch_size": Int(),
        "shuffle": Bool(),
        "num_workers": Int(),
    }
)

train_params = Map({"checkpoint_every_n_steps": Int(), "max_num_steps": Int(), "eval_every_n_steps": Int()})

schema_v4 = Map(
    {
        "format_version": Int(),
        "model_params": model_params,
        "raster_params": raster_params,
        "train_data_loader": data_loader,
        "val_data_loader": data_loader,
        "train_params": train_params,
    }
)


SCHEMA_FORMAT_VERSION_TO_SCHEMA = {4: schema_v4}


def load_config_data(path: str) -> dict:
    yaml_string = Path(path).read_text()
    cfg_without_schema = load(yaml_string, schema=None)
    schema_version = int(cfg_without_schema["format_version"])
    if schema_version not in SCHEMA_FORMAT_VERSION_TO_SCHEMA:
        raise Exception(f"Unsupported schema format version: {schema_version}.")

    strict_cfg = load(yaml_string, schema=SCHEMA_FORMAT_VERSION_TO_SCHEMA[schema_version])
    cfg: dict = strict_cfg.data
    return cfg


def config_data_to_config(data):  # type: ignore
    return as_document(data, schema_v4)


def save_config_data(data: dict, path: str) -> None:
    cfg_document = config_data_to_config(data)
    with open(Path(path), "w") as f:
        f.write(cfg_document.as_yaml())
