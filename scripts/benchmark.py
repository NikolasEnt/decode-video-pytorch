import os
import time

import numpy as np
import torch

from src.video_io import VALIVideoReader, OpenCVVideoReader, \
    AbstractVideoReader, PyNvVideoCodecReader, TorchcodecVideoReader, \
    TorchvisionVideoReader

# Provide your test video files here.
VIDEO_PATHS = [
    "/workdir/data/videos/test.mp4",
]
N_PASSES = 3  # Number of times to repeat the benchmark for each video

# Make sure the frames exist in the video file and all frames fit into VRAM!
FRAMES_TO_READ_SEQUENTIAL = list(range(10, 20))
FRAMES_TO_READ_SLICE = list(range(10, 200, 20))

# Define the video readers to test
VIDEO_READERS = [
    TorchcodecVideoReader,
    OpenCVVideoReader,
    TorchvisionVideoReader,
    VALIVideoReader,
    PyNvVideoCodecReader
]

MODES_TO_USE = ["seek", "stream"]
# Note that some of the video readers don't support 'cpu'
DEVICE = "cuda:0"

if DEVICE.startswith("cuda:"):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE.split(":")[1])
    DEVICE = "cuda:0"  # Since other devices are invisible now


def run_benchmark(video_reader: AbstractVideoReader,
                  frames_to_read: list[int] = FRAMES_TO_READ_SEQUENTIAL)\
        -> list[float]:
    """Run a benchmark for reading a video using a specific video reader class.

    Args:
        video_reader (AbstractVideoReader): The video reader to profile.
        frames_to_read (list[int]): List of frame indices to read.

    Returns:
        list[float]: Lists containing all individual reads time.
    """
    reading_time = []

    assert max(frames_to_read) < video_reader.num_frames, \
        (f"Frame index {max(frames_to_read)} is out of range "
         f"for video {video_reader.video_path}")

    _ = video_reader[0]  # Warmup

    for _ in range(N_PASSES):
        start_time = time.perf_counter()
        _ = video_reader.read_frames(frames_to_read)
        torch.cuda.synchronize()  # Ensure all CUDA operations are completed
        end_time = time.perf_counter()
        reading_time.append(end_time - start_time)

    return reading_time


def aggregate_results(timings: list[float]) -> tuple[float, float]:
    """Aggregate benchmark results from timings.

    Args:
        timings (list[float]): List of individual times.

    Returns:
        tuple[float, float]: Mean and std of the times.
    """
    mean_time = float(np.mean(timings))
    std_dev = float(np.std(timings))

    return mean_time, std_dev


def main():
    all_results = {
        reader_class.__name__: {mode: {'sequential': [], 'slice': []}
                                for mode in MODES_TO_USE}
        for reader_class in VIDEO_READERS
    }

    for video_path in VIDEO_PATHS:
        print(f"\nBenchmarking on video: {video_path}")
        for video_reader_class in VIDEO_READERS:
            for mode in MODES_TO_USE:
                print(f" - {video_reader_class.__name__}: {mode} mode")
                reader = video_reader_class(
                    video_path=video_path,
                    mode=mode,
                    device=DEVICE)

                timings_sequential = run_benchmark(
                    reader, FRAMES_TO_READ_SEQUENTIAL)
                print(
                    f"   Sequential frame reading times: {timings_sequential}")
                all_results[video_reader_class.__name__][mode]['sequential'].extend(
                    timings_sequential)

                timings_slice = run_benchmark(reader, FRAMES_TO_READ_SLICE)
                print(f"   Slice frame reading times: {timings_slice}")
                all_results[video_reader_class.__name__][mode]['slice'].extend(
                    timings_slice)

    for reader_name in all_results:
        for mode in MODES_TO_USE:
            sequential_mean, sequential_std_dev = aggregate_results(
                all_results[reader_name][mode]['sequential'])
            print(f"Final result for {reader_name} ({mode} mode) - Sequential:"
                  f" {sequential_mean:.4f} ± {sequential_std_dev:.4f} s")

            slice_mean, slice_std_dev = aggregate_results(
                all_results[reader_name][mode]['slice'])
            print(f"Final result for {reader_name} ({mode} mode) - Slice:"
                  f" {slice_mean:.4f} ± {slice_std_dev:.4f} s")


if __name__ == "__main__":
    main()
