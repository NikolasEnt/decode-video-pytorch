from typing import Literal
from pathlib import Path

import torch
import PyNvVideoCodec as nvc  # noqa: N813

from src.video_io.abstract_reader import AbstractVideoReader


class PyNvVideoCodecReader(AbstractVideoReader):
    def __init__(self,
                 video_path: str | Path,
                 mode: Literal["seek", "stream"] = "stream",
                 output_format: Literal["THWC", "TCHW"] = "THWC",
                 device: str = "cuda:0"):
        self.decoder = None
        super().__init__(video_path, mode, output_format, device)
        if self.gpu_id < 0:
            ValueError("PyNvVideoCodecReader supports Nvidia GPU decoding only"
                       f"provide a valid device. {self.device} was specified.")

    def _initialize_reader(self) -> None:
        self.decoder = nvc.SimpleDecoder(
            enc_file_path=self.video_path,
            gpu_id=self.gpu_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGB
        )
        metadata = self.decoder.get_stream_metadata()
        self.num_frames = metadata.num_frames
        # Docs suggests it's avg_frame_rate, but it does not exist
        self.fps = metadata.average_fps

    def seek_read(self, frame_indices: list[int]) -> list[torch.Tensor]:
        frames = []
        for idx in frame_indices:
            frames.append(torch.from_dlpack(self.decoder[idx]))
        return frames

    def stream_read(self, frame_indices: list[int]) -> list[torch.Tensor]:
        return [torch.from_dlpack(frame) for frame in
                self.decoder.get_batch_frames_by_index(frame_indices)]

    def release(self) -> None:
        if hasattr(self, 'decoder') and self.decoder is not None:
            del self.decoder
            self.decoder = None
