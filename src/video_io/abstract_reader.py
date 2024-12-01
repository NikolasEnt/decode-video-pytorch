import os
from abc import ABCMeta, abstractmethod
from typing import Any, Literal
from pathlib import Path

import numpy as np
import torch


class AbstractVideoReader(metaclass=ABCMeta):
    def __init__(self, video_path: str | Path,
                 mode: Literal["seek", "stream"] = "stream",
                 output_format: Literal["THWC", "TCHW"] = "THWC",
                 device: str = "cuda:0") -> None:

        if isinstance(video_path, Path):
            video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} does not exist")
        if mode not in ["seek", "stream"]:
            raise ValueError(f"Invalid mode {mode}. "
                             "Must be one of ['seek', 'stream']")
        if output_format not in ["THWC", "TCHW"]:
            raise ValueError(f"Invalid output_format {format}. "
                             "Must be one of ['THWC', 'TCHW']")
        self.video_path = video_path
        self.mode = mode
        self.output_format = output_format
        self.device = device
        # Values to be initialised
        self.num_frames: int = 0  # Number of frames in the video
        self.fps: int | float = 0
        self._initialize_reader()

    @abstractmethod
    def _initialize_reader(self) -> None:
        """Initialise metadata and prepare reader for reading the video."""
        pass

    @abstractmethod
    def _to_tensor(self, frames: Any) -> torch.Tensor:
        return torch.tensor(frames, device=self.device)

    def _process_frame(self, frame: np.ndarray | torch.Tensor)\
            -> np.ndarray | torch.Tensor:
        """Process an individual frame stacking."""
        return frame

    @abstractmethod
    def seek_read(self, frame_indices: list[int]) -> list[np.ndarray]:
        """Seek to each frame and read the frames from the video one by one."""
        pass

    @abstractmethod
    def stream_read(self, frame_indices: list[int]) -> list[np.ndarray]:
        """Read all frames in range of the given indices and subset them.

        Args:
            frame_indices (list[int]): List of frame indices to read. Indices
                are expected to be sorted, it is expected to be at least one
                index in the list.

        Returns:
           list[np.ndarray]: List of frames from the video.

        """
        pass

    def read_frames(self, frame_indices: list[int]) -> torch.Tensor:
        if min(frame_indices) < 0 or max(frame_indices) >= self.num_frames:
            raise ValueError(f"Invalid frame indices {frame_indices} "
                             f"in {self.video_path} video. "
                             f"Must be in range [0, {self.num_frames - 1}]")
        frames = []
        if self.mode == "seek":
            frames = self.seek_read(frame_indices)
        elif self.mode == "stream":
            frames = self.stream_read(frame_indices)
        return self._to_tensor(frames)

    @abstractmethod
    def release(self) -> None:
        """Release any resources used by the reader."""
        pass

    def __len__(self):
        return self.num_frames

    def _read_frames_slice(self, start_idx: int, stop_idx: int,
                           step: int) -> torch.Tensor:
        indices = list(range(start_idx, stop_idx, step))
        return self.read_frames(indices)

    def __getitem__(self, index: int | slice) -> torch.Tensor:
        if isinstance(index, int):
            if index < 0 or index >= self.num_frames:
                raise IndexError(
                    f"Index {index} is out of bounds for video "
                    f"{self.video_path} with {self.num_frames} frames.")
            return self.read_frames([index])

        if isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            start = start if start is not None else 0
            stop = stop if stop is not None else self.num_frames
            step = step if step is not None else 1

            if start < 0 or stop > self.num_frames or step <= 0:
                raise ValueError(f"Invalid slice {index} for video "
                                 f"with {self.num_frames} frames.")

            return self._read_frames_slice(start_idx=start, stop_idx=stop,
                                           step=step)

        raise TypeError(
            f"Index must be an integer or slice, not {type(index)}")

    def __repr__(self) -> str:
        return f"Video {self.video_path} {self.num_frames}frames@{self.fps}fps"
