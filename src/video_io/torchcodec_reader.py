from typing import List, Literal
from pathlib import Path

import torch
from torchcodec.decoders import VideoDecoder

from src.video_io.abstract_reader import AbstractVideoReader


class TorchcodecVideoReader(AbstractVideoReader):
    """Videoreader using TorchCodec library.

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
        dimension_order = "NHWC" if self.output_format == 'THWC' else 'NCHW'
        self.decoder = VideoDecoder(self.video_path,
                                    dimension_order=dimension_order,
                                    device=self.device)
        self.num_frames = self.decoder.metadata.num_frames
        self.fps = self.decoder.metadata.average_fps

    def _to_tensor(self, frames: torch.Tensor) -> torch.Tensor:
        return frames.to(self.device)

    def seek_read(self, frame_indices: List[int]) -> torch.Tensor:
        """Retrieve frames by their indices using random access."""
        frames = []
        for idx in frame_indices:
            if idx < 0 or idx >= self.num_frames:
                raise ValueError(f"Invalid frame index: {idx}")
            frame = self.decoder[idx]
            frames.append(self._process_frame(frame))
        return torch.stack(frames, dim=0)

    def stream_read(self, frame_indices: List[int]) -> torch.Tensor:
        start_idx, end_idx = min(frame_indices), max(frame_indices)
        if start_idx < 0 or end_idx >= self.num_frames:
            raise ValueError(f"Invalid frame indices: {frame_indices}")
        batch = self.decoder.get_frames_in_range(
            start=start_idx, stop=end_idx + 1, step=1)
        frames = batch.data  # Frames in TCHW format
        relative_indices = [idx - start_idx for idx in frame_indices]
        selected_frames = frames[relative_indices]  # Subset selection
        return self._to_tensor(selected_frames)

    def _read_frames_slice(self, start_idx: int, stop_idx: int, step: int)\
            -> torch.Tensor:
        return self.decoder[start_idx:stop_idx:step]

    def release(self) -> None:
        self.decoder = None
