def get_history_slice(
        frame_index: int, history_num_states: int, history_step_size: int, include_current_state: bool = False
) -> slice:
    """Given a frame index and history settings returns a slice that returns the given data in the right order.
    Note that this history returned starts with the most "recent" frame first (i.e. reverse in time as it's history).

    Example:
    ``frame_index=20``, ``history_num_frames=2``, ``history_step_size=2``, ``include_current_state=True``
    would return a slice for frame index 20, 18, 16.

    Arguments:
        state_index (int): The "anchor" frame index you want to sample from
        history_num_states (int): Number of history frames (not including the current frame).
        history_step_size (int): How many frames to step for each history step.

    Keyword Arguments:
        include_current_state (bool): Whether the slice should include ``frame_index`` (default: {False})

    Raises:
        IndexError: Returned when ``history_step_size`` is an invalid value (e.g. 0).

    Returns:
        slice -- Slice that when applied to an array returns the history frames in the right order.
    """
    if history_step_size <= 0:
        raise IndexError("History step size can not be 0 or negative")
    history_earliest_index = frame_index - (history_num_states) * history_step_size - 1
    history_latest_index = frame_index - (0 if include_current_state else history_step_size)

    if history_latest_index < 0:
        return slice(0, 0, -history_step_size)

    # Necessary for including the first element with negative step size (step size has to be negated as it's history)
    # Example: start index = 2, step size = 2, element 0 should be included.
    # +1 is required because it is non-inclusive (-1 is performed above).
    if history_earliest_index < 0 and (history_earliest_index + 1) % history_step_size == 0:
        history_earliest_index = None  # type: ignore
    else:
        history_earliest_index = max(0, history_earliest_index)

    return slice(history_latest_index, history_earliest_index, -history_step_size)


def get_future_slice(frame_index: int, future_num_states: int, future_step_size: int) -> slice:
    """Given a frame index and future settings returns a slice that returns the given data in the right order.
    Note that this history returned starts with the most "recent" frame first
    (e.g. ``current_frame``+``future_step_size``).

    Example:
    ``frame_index=20``, ``future_num_states=2``, ``future_step_size=2``
    would return a slice for frame index 22, 24.

    Arguments:
        state_index (int): The "anchor" frame index you want to sample from
        future_num_states (int): Number of future frames.
        future_step_size (int): How many frames to step for each future step.

    Raises:
        IndexError: Returned when ``future_step_size`` is an invalid value (e.g. 0).

    Returns:
        slice -- Slice that when applied to an array returns the future frames in the right order.
    """
    if future_step_size <= 0:
        raise IndexError("Future step size can not be 0")

    future_latest_index = frame_index + (future_num_states) * future_step_size + 1
    future_earliest_index = frame_index + future_step_size
    return slice(future_earliest_index, future_latest_index, future_step_size)
