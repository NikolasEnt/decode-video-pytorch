import os
from abc import ABCMeta, abstractmethod
from typing import Any, Literal
from pathlib import Path

import torch

from src.video_io.utils import get_device_id


class AbstractVideoReader(metaclass=ABCMeta):
    def __init__(self, video_path: str | Path,
                 mode: Literal["seek", "stream"] = "stream",
                 output_format: Literal["THWC", "TCHW"] = "THWC",
                 device: str = "cuda:0") -> None:
        self.video_path = self._validate_and_convert_path(video_path)
        self.mode = self._validate_mode(mode)
        self.output_format = self._validate_output_format(output_format)
        self.device = device
        self.gpu_id = get_device_id(device)

        # Values to be initialised
        self.num_frames: int = 0
        self.fps: int | float = 0
        self._initialize_reader()

    def _validate_and_convert_path(self, video_path: str | Path) -> str:
        if isinstance(video_path, Path):
            video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} does not exist")
        return video_path

    def _validate_mode(self, mode: Literal["seek", "stream"]) -> str:
        if mode not in ["seek", "stream"]:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of ['seek', 'stream']")
        return mode

    def _validate_output_format(
            self, output_format: Literal["THWC", "TCHW"]) -> str:
        if output_format not in ["THWC", "TCHW"]:
            raise ValueError(
                f"Invalid output format '{output_format}'. "
                "Must be one of ['THWC', 'TCHW']")
        return output_format

    @abstractmethod
    def _initialize_reader(self) -> None:
        """Initialise metadata and prepare reader for reading the video."""
        pass

    def _finalize_tensor(
            self, frames: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Combine frame tensors and finalize the output format.

        Args:
            frames (list[torch.Tensor] | torch.Tensor): A list of frame tensors
                to be combined or a single tensor of shape THWC.

        Returns:
            torch.Tensor: The unified and finalized tensor.
        """
        if isinstance(frames, list):
            tensor = torch.stack(frames, dim=0)
        else:
            tensor = frames
        tensor = tensor.to(self.device)
        if self.output_format == "TCHW":
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor

    def _process_frame(self, frame: Any) -> torch.Tensor:
        """Process an individual frame if required and convert it to tensor."""
        return frame

    @abstractmethod
    def seek_read(self, frame_indices: list[int]) -> list[torch.Tensor]:
        """Seek to each frame and read the frames from the video one by one."""
        pass

    @abstractmethod
    def stream_read(self, frame_indices: list[int]) -> list[torch.Tensor]:
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
        return self._finalize_tensor(frames)

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
        return (f"Video {self.video_path}: "
                f"{self.num_frames} frames @ {self.fps}fps")
