"""This is an example how to use PyNvVideoCodec with PyTorch.

The currently available version of PyNvVideoCodec has some issues with Seek()
command, so it is challenging to provide random frame access. Therefore,
this example uses a sequential approach to decode the video frames.
"""

import torch
import PyNvVideoCodec as nvc  # noqa

# Adjust the video path to your test video.
nv_dmx = nvc.CreateDemuxer(filename="/workdir/data/videos/test.mp4")
nv_dec = nvc.CreateDecoder(gpuid=0,
                           codec=nv_dmx.GetNvCodecId(),
                           cudacontext=0,
                           cudastream=0,
                           usedevicememory=True)

for packet in nv_dmx:
    for decoded_frame in nv_dec.Decode(packet):
        tensor = torch.from_dlpack(decoded_frame)  # NV12
        print(tensor.shape, tensor.dtype, tensor.device)
