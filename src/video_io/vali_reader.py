import warnings
from typing import Literal
from pathlib import Path

import torch
import python_vali as vali

from src.video_io.abstract_reader import AbstractVideoReader


class VALIVideoReader(AbstractVideoReader):
    """Videoreader using VALI.

    See details on VALI at https://github.com/RomanArzumanyan/VALI.

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
        self.device_id = -1  # CPU decoder by default
        if device.startswith("cuda:"):
            self.device_id = int(device.split(":")[1])
        elif device == "cuda":
            self.device_id = 0
        elif device == "cpu":
            self.device_id = -1
        else:
            warnings.warn(f"Unknown device {device}, using CPU instead.",
                          stacklevel=2)
        super().__init__(video_path, mode=mode, output_format=output_format,
                         device=device)

    def _initialize_reader(self) -> None:
        self._decoder = vali.PyDecoder(self.video_path, opts={},
                                       gpu_id=self.device_id)
        self.num_frames = self._decoder.NumFrames
        self.width = self._decoder.Width
        self.height = self._decoder.Height
        self.fps = self._decoder.AvgFramerate

        # NV12 -> RGB conversion. Feel free to adjust as needed
        target_format = vali.PixelFormat.RGB
        self._nv12_to_rgb = vali.PySurfaceConverter(gpu_id=self.device_id)

        self.surf_nv12 = vali.Surface.Make(
            format=vali.PixelFormat.NV12, width=self.width, height=self.height,
            gpu_id=self.device_id)

        self.surf_rgb = vali.Surface.Make(
            format=target_format, width=self.width, height=self.height,
            gpu_id=self.device_id)

        # Note, some video containers may not have this information
        self._cc_ctx = vali.ColorspaceConversionContext(
            self._decoder.ColorSpace, self._decoder.ColorRange
        )

    def _decode_surface(self, surface: vali.Surface) -> torch.Tensor:
        self._nv12_to_rgb.Run(surface, self.surf_rgb, cc_ctx=self._cc_ctx)
        frame_tensor = torch.from_dlpack(self.surf_rgb)
        frame_tensor = frame_tensor.clone().detach()
        return frame_tensor

    def _to_tensor(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.to(self.device)
        if self.output_format == "TCHW":
            frames = frames.permute(0, 3, 1, 2)
        return frames

    def seek_read(self, frame_indices: list[int]) -> list[torch.Tensor]:
        frame_tensors = []
        for idx in frame_indices:
            seek_ctx = vali.SeekContext(idx)
            success, details = self._decoder.DecodeSingleSurface(
                self.surf_nv12, seek_ctx=seek_ctx)
            if not success:
                raise RuntimeError(f"Failed to decode frame {idx}: {details}")
            frame_tensors.append(self._decode_surface(self.surf_nv12))
        tensor = torch.stack(frame_tensors, dim=0)
        return tensor

    def stream_read(self, frame_indices: list[int]) -> torch.Tensor:
        start_idx = frame_indices[0]  # Assuming the indices are sorted
        seek_ctx = vali.SeekContext(start_idx)
        success, details = self._decoder.DecodeSingleSurface(
            self.surf_nv12, seek_ctx=seek_ctx)
        if not success:
            raise RuntimeError(
                f"Failed to decode frame {start_idx}: {details}")
        frame_tensors = [self._decode_surface(self.surf_nv12)]
        for idx in range(start_idx, max(frame_indices)):
            success, details = self._decoder.DecodeSingleSurface(
                self.surf_nv12)
            if not success:
                raise RuntimeError(f"Failed to decode frame {idx}: {details}")
            if idx in frame_indices:
                frame_tensors.append(self._decode_surface(self.surf_nv12))
        tensor = torch.stack(frame_tensors, dim=0)
        return tensor

    def release(self) -> None:
        del self._decoder
        del self.surf_nv12
        del self.surf_rgb
        del self._cc_ctx
