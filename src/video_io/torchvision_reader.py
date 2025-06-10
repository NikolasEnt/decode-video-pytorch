
from typing import Literal
from pathlib import Path

import torch
from torchvision.io import read_video, read_video_timestamps

from src.video_io.abstract_reader import AbstractVideoReader


class TorchvisionVideoReader(AbstractVideoReader):
    """Videoreader using PyTorch's torchvision.io.read_video.

    Note:
        TorchVision video decoding and encoding will be removed in a future
        release of TorchVision, TorchCodec is a recommended approach.

    Args:
        video_path (str | Path): Path to the input video file.
        mode (Literal["seek", "stream"], optional): Reading mode: "seek" -
            find each frame individually, "stream" - decode all frames from
            the range of requested indeces and subsample.
            Defaults to "stream".
        output_format (Literal["THWC", "TCHW"], optional): Data format:
            channel last or first. Defaults to "THWC".
        device (str, optional): Device to send the resulted tensor to.
            Defaults to "cuda:0".
    """

    def __init__(self, video_path: str | Path,
                 mode: Literal["seek", "stream"] = "stream",
                 output_format: Literal["THWC", "TCHW"] = "THWC",
                 device: str = "cuda:0"):
        super().__init__(video_path, mode=mode, output_format=output_format,
                         device=device)

    def _initialize_reader(self) -> None:
        timestamps, fps = read_video_timestamps(self.video_path,
                                                pts_unit="sec")
        self.timestamps = timestamps
        self.num_frames = len(timestamps)
        self.fps = fps

    def seek_read(self, frame_indices: list[int]) -> list[torch.Tensor]:
        frame_timestamps = [self.timestamps[fid] for fid in frame_indices]
        frames = []
        for ts in frame_timestamps:
            frame, _, _ = read_video(
                self.video_path, start_pts=ts, end_pts=ts,
                pts_unit="sec", output_format="THWC")
            frames.append(self._process_frame(frame[0]))
        return frames

    def stream_read(self, frame_indices: list[int]) -> torch.Tensor:
        frame_timestamps = [self.timestamps[fid] for fid in frame_indices]
        frames, _, _ = read_video(
            self.video_path, start_pts=min(frame_timestamps),
            end_pts=max(frame_timestamps) + (1 / self.fps),
            pts_unit="sec", output_format="THWC")
        frame_indices_sample = [fid - min(frame_indices)
                                for fid in frame_indices]
        frames = frames[frame_indices_sample]
        return frames

    def release(self) -> None:
        pass
