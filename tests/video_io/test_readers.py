import os

import torch
import pytest

from src.video_io import VALIVideoReader, OpenCVVideoReader, \
    PyNvVideoCodecReader, TorchcodecVideoReader, TorchvisionVideoReader

VIDEO_PATH = '/workdir/data/videos/test.mp4'
FRAMES_TO_READ = [10, 20, 30, 40]
DEVICE = "cuda:0"

VIDEO_READERS = [
    TorchcodecVideoReader,
    OpenCVVideoReader,
    TorchvisionVideoReader,
    VALIVideoReader,
    PyNvVideoCodecReader
]

MODES = ["seek", "stream"]
OUTPUT_FORMATS = {"THWC": (4, None, None, 3), "TCHW": (4, 3, None, None)}

TORCH_DEVICE = torch.device(DEVICE)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA is not available")
@pytest.mark.skipif(not os.path.exists(VIDEO_PATH),
                    reason=(f"Video file {VIDEO_PATH} does not exist, "
                            "please provide a test video file"))
@pytest.mark.parametrize("reader_class", VIDEO_READERS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("output_format", OUTPUT_FORMATS.keys())
def test_video_reader(reader_class, mode, output_format):
    reader = reader_class(
        video_path=VIDEO_PATH,
        mode=mode,
        output_format=output_format,
        device=DEVICE
    )

    frames = reader.read_frames(FRAMES_TO_READ)
    expected_shape = OUTPUT_FORMATS[output_format]

    assert len(frames.shape) == len(expected_shape), \
        f"Expected {len(expected_shape)} dims, but got {len(frames.shape)}"

    for i, (exp_dim, actual_dim) in enumerate(
            zip(expected_shape, frames.shape, strict=True)):
        if exp_dim is not None:
            assert exp_dim == actual_dim, \
                (f"Dimension mismatch at position {i}: "
                 f"expected {exp_dim}, got {actual_dim}")

    assert frames.device == TORCH_DEVICE, \
        f"Expected device {DEVICE}, but got {frames.device}"


MAX_AVG_PIXEL_DIFF = 6.0


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA is not available")
@pytest.mark.skipif(not os.path.exists(VIDEO_PATH),
                    reason=(f"Video file {VIDEO_PATH} does not exist, "
                            "please provide a test video file"))
@pytest.mark.parametrize("reader_class", VIDEO_READERS)
@pytest.mark.parametrize("output_format", OUTPUT_FORMATS.keys())
def test_modes_equality(reader_class, output_format):
    reader_seek = reader_class(
        video_path=VIDEO_PATH,
        mode="seek",
        output_format=output_format,
        device=DEVICE
    )

    reader_stream = reader_class(
        video_path=VIDEO_PATH,
        mode="stream",
        output_format=output_format,
        device=DEVICE
    )

    frames_seek = reader_seek.read_frames(FRAMES_TO_READ).float()
    frames_stream = reader_stream.read_frames(FRAMES_TO_READ).float()
    m_err = torch.abs(frames_seek - frames_stream).mean().item()
    assert m_err < MAX_AVG_PIXEL_DIFF, \
        ("Frames by 'seek' mode should be roughly equal to those by "
         f"'stream' mode. Avg pixel diff {m_err:.2f} > {MAX_AVG_PIXEL_DIFF}.")
