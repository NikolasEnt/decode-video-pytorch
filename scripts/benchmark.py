import time

import numpy as np
import torch

from src.video_io import OpenCVVideoReader, AbstractVideoReader, \
    TorchvisionVideoReader

VIDEO_PATH = "/workdir/data/videos/test.mp4"
N_PASSES = 3  # Number of times to repeat the benchmark for each video
# Make sure the frames exists in video file and all frames fit into VRAM!
FRAMES_TO_READ_SEQUENTIAL = list(range(10, 20))
FRAMES_TO_READ_SLICE = list(range(10, 200, 10))

# Adjust the list of video readers to benchmark as needed
VIDEO_READERS = [
    OpenCVVideoReader,
    TorchvisionVideoReader,
]
MODES_TO_USE = ["seek", "stream"]
DEVICE = "cuda:0"  # Device to use for the benchmark


def run_benchmark(video_reader: AbstractVideoReader,
                  frames_to_read: list[int] = FRAMES_TO_READ_SEQUENTIAL)\
        -> tuple[float, float]:
    """Run a benchmark for reading a video using a specific video reader class.

    Args:
        video_reader (AbstractVideoReader): The video reader to profile.
        frames_to_read (list[int]): List of frame indices to read.

    Returns:
        float: Average time taken to read the video in seconds.
        float: Standard deviation of the time.
    """
    reading_time = []
    assert max(frames_to_read) < video_reader.num_frames, \
        (f"Frame index {max(frames_to_read)} is out of range "
         f"for video {video_reader.video_path}")
    for _ in range(N_PASSES):
        start_time = time.perf_counter()
        _ = video_reader.read_frames(frames_to_read)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        reading_time.append(end_time - start_time)
    return float(np.mean(reading_time)), float(np.std(reading_time))


def main():
    for video_reader_class in VIDEO_READERS:
        for mode in MODES_TO_USE:
            print(f"Benchmarking {video_reader_class.__name__}: {mode}")
            reader = video_reader_class(
                video_path=VIDEO_PATH,
                mode=mode,
                device=DEVICE)
            mean_time, std_dev = run_benchmark(
                reader, FRAMES_TO_READ_SEQUENTIAL)
            print("Mean time of sequential frames read: "
                  f"{mean_time:.4f} ± {std_dev:.4f} s")
            mean_time, std_dev = run_benchmark(
                reader, FRAMES_TO_READ_SLICE)
            print("Mean time of random frames read: "
                  f"{mean_time:.4f} ± {std_dev:.4f} s")


if __name__ == "__main__":
    main()
