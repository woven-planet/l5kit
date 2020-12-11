import numpy as np
import pytest
import torch

from l5kit.dataset.utils import convert_str_to_fixed_length_tensor, kMaxStrLength


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
