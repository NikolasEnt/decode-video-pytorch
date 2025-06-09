import warnings

from torch.cuda import device_count, is_available


def get_device_id(device: str) -> int:
    """Convert device name to device ID.

    Cpu decoded to -1.

    Args:
        device (str): The device name ("cpu", "cuda", or "cuda:<id>").

    Returns:
        int: The device ID.

    Raises:
        ValueError: If the requested CUDA device ID is not available.
    """
    if device.startswith("cuda:"):
        try:
            device_id = int(device.split(":")[1])
            if device_id < 0 or device_id >= device_count():
                raise ValueError(
                    f"CUDA device {device_id} does not exist. "
                    f"Available devices: {list(range(device_count()))}")
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid CUDA device format: {device}") from e
    elif device == "cuda":
        if is_available():
            device_id = 0
        else:
            warnings.warn(
                "CUDA is not available. Using CPU instead.", stacklevel=2)
            return -1
    elif device == "cpu":
        device_id = -1
    else:
        warnings.warn(f"Unknown device {device}, using CPU instead.",
                      stacklevel=2)
        device_id = -1

    return device_id
