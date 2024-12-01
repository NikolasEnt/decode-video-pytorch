# Video decoding for DL models training with PyTorch

The repo was originally developed to illustrate a talk given at the [London PyTorch Meetup](https://www.meetup.com/London-PyTorch-Meetup/):
<h5 align="center">
  Optimising Video Pipelines for Neural Network Training with PyTorch<br>
      by <i>Nikolay Falaleev</i>  on 21/11/2024
</h5>

The talk's slides are available [here](https://docs.google.com/presentation/d/1Qw9Cy0Pjikf5IBdZIGVqK968cKepKN2GuZD6hA1At8s/edit?usp=sharing).

It containes examples of different approaches to video frames decoding, which can be used
for training deep learning models with PyTorch.

## Prerequisites

* Nvidia GPU with Video Encode and Decode feature [CUVID](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new). Nvidia Driver version >= 535.
* GNU [make](https://www.gnu.org/software/make/) - it is quite likely that it is already installed on your system.
* [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
* Some video files for testing, put them in `/data/videos` directory.

## How to run

The project is provided with a Docker environment with PyTorch as well as FFmpeg and OpenCV compiled with Nvidia HW acceleration support.

1. Build Docker image:

```
make build
```

2. Run the container:
```
make run
```

All the following can be executed inside the running container.

## Code navigation

Several base video readers classes are provided in [src/video_io](src/video_io)]; they follow the same interface and inherit from [AbstractVideoReader](src/video_io/abstract_reader.py).

* [OpenCVVideoReader](src/video_io/opencv_reader.py) - Uses OpenCV's `cv2.VideoCapture` with the FFmpeg backend. It is the most straightforward way to read videos. Use `os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid"` to enable hardware acceleration. Adjust the video codec `h264_cuvid` parameter to match your video codec, e.g. `h264_cuvid` for h.264 codec and `hevc_cuvid` for HEVC codec; see all available codecs with Nvidia HW acceleration `ffmpeg -decoders | grep -i nvidia`.
* [TorchvisionReadVideo](src/video_io/torchvision_reader.py) - uses PyTorch's `torchvision.io` module.
* [TorchcodecVideoReader](src/video_io/torchcodec_reader.py) - uses [TorchCodec](https://github.com/pytorch/torchcodec) library. As TorchCodec is still in early stages of development and is installed from nightly builds, it may not work at some point or the API may change. This is likely to be the fastest vide reader in the project.
* [VALIVideoReader](src/video_io/vali_reader.py) - uses [VALI](https://github.com/RomanArzumanyan/VALI) library, which is a continuation of the [VideoProcessingFramework](https://github.com/NVIDIA/VideoProcessingFramework) project, which was discontinued by Nvidia. Unlike [PyNvVideoCodec](https://pypi.org/project/PyNvVideoCodec/), which is the current substitution by Nvidia, VALI offers a more flexible solution that includes pixel format and color space conversion capabilities, as well as some low-level operations on surfaces. This allows it to be more powerful than PyNvVideoCodec, although it has a steeper learning curve, VALI allows for building more complex and optimized pipelines.

A simple benchmark script is provided in [scripts/benchmark.py](src/scripts/benchmark.py). It compares the performance of different readers. Adjust parameters of the benchmark as required. To run the script,run the following command in the project container:

```bash
python scripts/benchmark.py
```


In addition, there are some other examples of video-related components in the project:
* [Kornia video augmentations](src/transforms.py) transforms.



### Try one of the video readers:

```python
from src.video_io import TorchvisionVideoReader

video_reader = TorchvisionVideoReader(
        "/workdir/data/videos/test.mp4", mode = "stream", output_format = "TCHW",
        device = "cuda:0")

frames_to_read = list(range(0, 100, 5))  # Read every 5th from
tensor = video_reader.read_frames(frames_to_read)
print(tensor.shape, tensor.device)  # Should be (20, 3, H, W), cuda:0
```

All video readers classes uses the same interface and return PyTorch tensors.

Arguments:

_video_path_ (str | Path): Path to the input video file.

_mode_ (`seek` or `stream`): Reading mode: `seek` -
find each frame individually, `stream` - decode all frames in
the range of requested indeces and subsample.
Defaults to `stream`.

_output_format_ (`THWC` or `TCHW`): Data format:
channel last or first. Defaults to `THWC`.

_device_ (str, optional): Device to send the resulted tensor to. If possible,
the same device will be used for HW acceleration of decoding. Defaults to `cuda:0`.

When using `mode = 'stream'`, one needs to ensure that all frames in the range (min(frames_to_read), max(frames_to_read)) fit into VRAM.
