# Video decoding for DL models training with PyTorch

This project demonstrates various approaches to decoding video frames into PyTorch tensors with hardware acceleration, providing benchmarks and examples to help users select the most efficient video reader for their deep learning workflows.

The repo was originally developed to illustrate a talk given at the [London PyTorch Meetup](https://www.meetup.com/London-PyTorch-Meetup/):
<h5 align="center">
  Optimising Video Pipelines for Neural Network Training with PyTorch<br>
      by <i>Nikolay Falaleev</i> on 21/11/2024
</h5>

The talk's slides are available [here](https://docs.google.com/presentation/d/1Qw9Cy0Pjikf5IBdZIGVqK968cKepKN2GuZD6hA1At8s/edit?usp=sharing) Note that the code was substantially updated since the talk's presentation, including new video readers and improvements in the code structure.

It contains examples of different approaches to decoding video frames directly into tensors, which can be used for training deep learning models with PyTorch.

![Benchmarks results](/readme_imgs/benchmarks.png)
_Time of video decoding into PyTorch tensors for different video readers in different modes. The reported values are for decoding 10 frames into PyTorch tensors from 1080p 30fps video file: [Big Buck Bunny](https://download.blender.org/demo/movies/BBB/). The results were obtained using Nvidia RTX 3090 for hardware acceleration of all decoders._

## Prerequisites

* Nvidia GPU with Video Encode and Decode feature [CUVID](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new). Nvidia Driver version >= 570.
* GNU [make](https://www.gnu.org/software/make/) - it is quite likely that it is already installed on your system.
* [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
* Some video files for testing, put them in the `data/videos` directory.

## How to run

The project is provided with a Docker environment that includes PyTorch, as well as FFmpeg and OpenCV, which are compiled from source with NVIDIA hardware acceleration support.

1. Build Docker image:

```
make build
```

2. Run the container:
```
make run
```

The Docker container will have the project folder mounted to `/workdir`, including the contents of `data` and all the code.

All the following can be executed inside the running container.

## Benchmarking
A simple benchmark script is provided in [scripts/benchmark.py](scripts/benchmark.py). It compares the performance of different readers available in the project running in different modes.

In order to run benchmarks, provide representative video files and update parameters of the benchmarking process in [scripts/benchmark.py](scripts/benchmark.py). Please note that the results heavily depend on video file features, including encoding parameters and resolution. Another critical aspect is the required sampling strategy - whether it is required to sample individual frames randomly, a sequence of frames or a sparse subset of frames. That is why it is recommended to run the benchmark with parameters representing the actual use case of the video reader to select the most appropriate one as well as select the best strategy for reading frames.

Adjust parameters of the benchmark as required. To run the script, run the following command in the project container:

```bash
python scripts/benchmark.py
```

When selecting a particular video decoding approach, one should consider additional features offered by the tools. For example, although VALI may not be the fastest in the provided benchmarking framework, it offers significant flexibility and can outperform other readers when additional transforms are required as part of the pipeline, such as colour space conversion, resizing, and more.

## Code navigation

Several base video readers classes are provided in [src/video_io](src/video_io); they follow the same interface and inherit from [AbstractVideoReader](src/video_io/abstract_reader.py).

* [OpenCVVideoReader](src/video_io/opencv_reader.py) - Uses OpenCV's `cv2.VideoCapture` with the FFmpeg backend. It is the most straightforward way to read videos. Use `os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid"` to enable hardware acceleration. Adjust the video codec `h264_cuvid` parameter to match your video codec, e.g. `h264_cuvid` for h.264 codec and `hevc_cuvid` for HEVC codec; see all available codecs with Nvidia HW acceleration `ffmpeg -decoders | grep -i nvidia`. The provided Docker image includes OpenCV with hardware acceleration enabled, as well as FFmpeg compiled with Nvidia components.
* [TorchvisionReadVideo](src/video_io/torchvision_reader.py) - uses PyTorch's `torchvision.io` module.
* [TorchcodecVideoReader](src/video_io/torchcodec_reader.py) - uses [TorchCodec](https://github.com/pytorch/torchcodec) library. As TorchCodec is still in early stages of development and is installed from nightly builds, it may not work at some point or the API may change, but it is the recommended native approach for PyTorch.
* [VALIVideoReader](src/video_io/vali_reader.py) - uses [VALI](https://github.com/RomanArzumanyan/VALI) library, which is a continuation of the [VideoProcessingFramework](https://github.com/NVIDIA/VideoProcessingFramework) project, which was discontinued by Nvidia. Unlike [PyNvVideoCodec](https://pypi.org/project/PyNvVideoCodec/), which is the current substitution by Nvidia, VALI offers a more flexible solution that includes pixel format and colour space conversion capabilities, as well as some low-level operations on surfaces. This allows it to be more powerful than PyNvVideoCodec, although it has a steeper learning curve, VALI allows for building more complex and optimized pipelines.
* [PyNvVideoCodecReader](src/video_io/nvcodec_reader.py) - uses [PyNvVideoCodec](https://developer.nvidia.com/pynvvideocodec) project by Nvidia. It is one of the highest-performing options for decoding videos on Nvidia GPUs. Documentation on PyNvVideoCodec can be found [here](https://docs.nvidia.com/video-technologies/pynvvideocodec/index.html).

In addition, there are some other examples of video-related components in the project:
* [Kornia video augmentations](src/transforms.py) transforms.


### Try one of the video readers:

```python
from src.video_io import TorchcodecVideoReader

video_reader = TorchcodecVideoReader(
    "/workdir/data/videos/test.mp4", mode = "stream", output_format = "TCHW",
    device = "cuda:0")

frames_to_read = list(range(0, 100, 5))  # Read every 5th frame
tensor = video_reader.read_frames(frames_to_read)
print(tensor.shape, tensor.device)  # Should be (20, 3, H, W), cuda:0
```

All video readers classes use the same interface and return PyTorch tensors.

Arguments:

_video_path_ (str or Path): Path to the input video file.

_mode_ (`seek` or `stream`): Reading mode: `seek` -
find each frame individually, `stream` - read all frames in
the range of requested indices (but not necessarily decode all frames) and subsample them. When using `mode = 'stream'`,
one needs to ensure that all frames in the range
(min(frames_to_read), max(frames_to_read)) fit into VRAM.
Defaults to `stream`.

_output_format_ (`THWC` or `TCHW`): Data format:
channels-last or channels-first. Defaults to `THWC`.

_device_ (str, optional): Device to send the resulted tensor to. If possible, the same device will be used for HW acceleration of decoding. Defaults to `cuda:0`.


## Known Limitations

* TorchVision video decoding and encoding features are deprecated and will be removed in a future release of TorchVision. TorchCodec is the recommended alternative and is actively being developed for native integration with PyTorch.
* The project currently does not implement asynchronous operations.
* As for now, RGB is supported as the only target colour space. The main purpose of the project is to provide a unified interface for different video readers for convenient testing and selection of the most suitable one. In real scenarios, one may need to further customise the functionality to support particular formats and transforms in the most optimal way to fit the requirements of specific use cases.
* The project is focused on Nvidia-based hardware acceleration, so `cpu` device is not properly supported and many readers are Nvidia-only.

## Acknowledgements

This project demonstrates the use of several great open-source libraries and frameworks:

- **[Torchcodec](https://github.com/pytorch/torchcodec)** – an experimental PyTorch library for video decoding, which is actively developed and offers promising native integration with PyTorch.
- **[VALI](https://github.com/RomanArzumanyan/VALI)** – a powerful and flexible video processing library, based on the discontinued NVIDIA Video Processing Framework. It provides low-level control and is particularly well-suited for complex hardware-accelerated pipelines, where some additional frame processing (colour space conversion, resizing, etc.) is required as part of the pilelien.
- **[PyNvVideoCodec](https://developer.nvidia.com/pynvvideocodec)** – an official NVIDIA project that provides Python bindings for video decoding using CUDA and NVDEC.
- **[OpenCV](https://opencv.org/)** – a widely-used computer vision library, with hardware-accelerated video decoding capabilities when compiled with FFmpeg and CUDA support.
- **[Kornia](https://kornia.org/)** – an open-source computer vision library for PyTorch, used in this project for video data augmentation examples.
- **[FFmpeg](https://ffmpeg.org/)**
