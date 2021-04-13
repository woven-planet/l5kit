import numpy as np
import pytest
import torch

from l5kit.dataset.utils import convert_str_to_fixed_length_tensor, kMaxStrLength, move_to_device, move_to_numpy


def test_convert_str() -> None:
    # assert the type of the return
    assert convert_str_to_fixed_length_tensor("test").dtype == torch.uint8

    # test with a string with the same value
    rep_count = 10
    fixed_str = "a" * rep_count
    str_cast = convert_str_to_fixed_length_tensor(fixed_str).numpy()
    assert len(np.unique(str_cast[:rep_count])) == 1
    assert np.allclose(str_cast[rep_count:], 0)

    # test with a str with different values
    fixed_str = "ab"
    str_cast = convert_str_to_fixed_length_tensor(fixed_str).numpy()
    assert len(np.unique(str_cast)) == 3

    # test with a str longer than th
    with pytest.raises(AssertionError):
        convert_str_to_fixed_length_tensor("a" * (kMaxStrLength + 1))


def test_move_to_numpy() -> None:
    in_dict = {"k1": torch.zeros(10), "k2": torch.ones(4)}
    for k in in_dict:
        assert isinstance(in_dict[k], torch.Tensor)

    out_dict = move_to_numpy(in_dict)
    assert np.alltrue(list(in_dict.keys()) == list(out_dict.keys()))
    for k in out_dict:
        assert isinstance(out_dict[k], np.ndarray)


def test_move_to_device_trivial() -> None:
    in_dict = {"k1": torch.zeros(10), "k2": torch.ones(4)}
    for k in in_dict:
        assert isinstance(in_dict[k], torch.Tensor)
        assert in_dict[k].device == torch.device("cpu")

    out_dict = move_to_device(in_dict, torch.device("cpu"))
    assert np.alltrue(list(in_dict.keys()) == list(out_dict.keys()))
    for k in out_dict:
        assert isinstance(out_dict[k], torch.Tensor)
        assert out_dict[k].device == torch.device("cpu")
