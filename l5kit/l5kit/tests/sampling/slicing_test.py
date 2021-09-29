from l5kit.sampling.slicing import get_future_slice, get_history_slice


def test_future_slice() -> None:
    # Current index 0, 2 future states with step size 3
    # Should yield indices 3 and 6, so slice (3, 7, 3)
    assert get_future_slice(0, 2, 3) == slice(3, 7, 3)
    assert get_future_slice(1, 2, 3) == slice(4, 8, 3)
    assert get_future_slice(11, 2, 3) == slice(14, 18, 3)


def test_history_slice() -> None:
    # Current index 10, 2 history states with step size 3
    # Should yield indices 4 and 7, so slice (7, 3, 3)
    assert get_history_slice(10, 2, 3) == slice(7, 3, -3)
    assert get_history_slice(10, 2, 3, include_current_state=True) == slice(10, 3, -3)

    assert get_history_slice(20, 2, 3) == slice(17, 13, -3)
    assert get_history_slice(20, 3, 3) == slice(17, 10, -3)
    assert get_history_slice(10, 2, 1) == slice(9, 7, -1)

    assert get_history_slice(20, 2, 3, include_current_state=True) == slice(20, 13, -3)
    assert get_history_slice(20, 3, 3, include_current_state=True) == slice(20, 10, -3)
    assert get_history_slice(10, 2, 1, include_current_state=True) == slice(10, 7, -1)

    # Not possible here to go past the first state, should give an empty slice
    # note range(10)[slice(0, 0, -3)] == []
    assert get_history_slice(1, 2, 3) == slice(0, 0, -3)
    assert get_history_slice(2, 2, 3) == slice(0, 0, -3)
    assert get_history_slice(0, 2, 3) == slice(0, 0, -3)

    # Partially possible here
    assert get_history_slice(3, 2, 3) == slice(0, None, -3)
