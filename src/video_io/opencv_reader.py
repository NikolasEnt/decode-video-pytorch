import os
from typing import Any, Literal
from pathlib import Path

import cv2
import numpy as np
import torch

from src.video_io.abstract_reader import AbstractVideoReader

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class OpenCVVideoReader(AbstractVideoReader):
    """OpenCV-based video reader.

    To enable Nvidia GPU decoding, add the following:

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU to use for decoding

    Adjust the codec 'h264_cuvid' in accordance to the input video file codec.

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
        self._cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        self.num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)

    def seek_read(self, frame_indices: list[int]) -> list[np.ndarray]:
        frames = []
        for idx in frame_indices:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self._cap.read()
            if ret:
                frames.append(self._process_frame(frame))
            else:
                break
        return frames

    def stream_read(self, frame_indices: list[int]) -> list[np.ndarray]:
        frames = []
        start_idx = min(frame_indices)
        end_idx = max(frame_indices) + 1
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for idx in range(start_idx, end_idx):
            ret, frame = self._cap.read()
            if not ret:
                break
            if idx in frame_indices:
                frames.append(self._process_frame(frame))
        return frames

    def _to_tensor(self, frames: Any) -> torch.Tensor:
        frames = torch.stack([torch.from_numpy(frame)
                              for frame in frames], dim=0).to(self.device)
        if self.output_format == "TCHW":
            frames = frames.permute(0, 3, 1, 2)
        return frames

    def release(self):
        self._cap.release()
