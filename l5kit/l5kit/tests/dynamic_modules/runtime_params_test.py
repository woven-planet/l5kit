from l5kit.dynamic_modules.loaders.runtime_params import RuntimeParams


def test_default_runtime_params() -> None:
    rp = RuntimeParams.get()
    assert "username" in rp
