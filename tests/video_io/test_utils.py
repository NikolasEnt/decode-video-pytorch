from unittest.mock import patch

import pytest

from src.video_io.utils import get_device_id


def mock_device_count():
    return 2


def mock_is_available_true():
    return True


def mock_is_available_false():
    return False


@pytest.mark.parametrize(
    "device_input, expected_output",
    [
        ("cpu", -1),
        ("cuda", 0),
        ("cuda:0", 0),
        ("cuda:1", 1),
    ]
)
@patch('src.video_io.utils.device_count', side_effect=mock_device_count)
@patch('src.video_io.utils.is_available', side_effect=mock_is_available_true)
def test_get_device_id_happy_cases(mock_is_available, mock_device_count,
                                   device_input, expected_output):
    assert get_device_id(device_input) == expected_output


@pytest.mark.parametrize(
    "device_input",
    [
        ("cuda:-1"),
        ("cuda:4"),
    ]
)
@patch('src.video_io.utils.device_count', side_effect=mock_device_count)
@patch('src.video_io.utils.is_available', side_effect=mock_is_available_true)
def test_get_device_id_error_cases(mock_is_available, mock_device_count,
                                   device_input):
    with pytest.raises(ValueError):
        get_device_id(device_input)


@pytest.mark.parametrize(
    "device_input, expected_output",
    [
        ("cpu", -1),
        ("cuda", -1),
    ]
)
@patch('src.video_io.utils.is_available', side_effect=mock_is_available_false)
def test_get_device_id_no_gpus(mock_is_available, device_input,
                               expected_output):
    assert get_device_id(device_input) == expected_output
