import torch

kMaxStrLength = 70  # Max string length for fixed length byte encoding.


def convert_str_to_fixed_length_tensor(string: str, max_length: int = kMaxStrLength) -> torch.Tensor:
    """
    Converts a string into a fixed length tensor of type torch.uint8.

    Args:
        string (str): String to convert
        max_length (int): Fixed length, default kMaxStrLength

    Returns:
        torch.Tensor: A fixed length tensor of type torch.uint8 that encodes the string
    """
    assert (
        len(string) <= max_length
    ), f"Encountered string longer that maximum length supported ({max_length}): {string}"
    assert "\0" not in string, f"String contains 0 value used for padding: {string}"
    return torch.cat(
        (
            torch.ByteTensor(torch.ByteStorage.from_buffer(string.encode("ascii"))),  # type: ignore
            torch.zeros(max_length - len(string), dtype=torch.uint8),
        )
    )
