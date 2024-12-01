from .vali_reader import VALIVideoReader
from .opencv_reader import OpenCVVideoReader
from .abstract_reader import AbstractVideoReader
# The current nightly version of TorchCodec does not work, at least in the env
# from .torchcodec_reader import TorchcodecVideoReader
from .torchvision_reader import TorchvisionVideoReader
