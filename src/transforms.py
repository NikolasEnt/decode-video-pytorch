"""An example of kornia-based video transforms.

See documentation in:
https://kornia.readthedocs.io/en/latest/augmentation.container.html#video-data-augmentation
"""
import kornia.augmentation as augm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transform = augm.VideoSequential(
    augm.Normalize(mean=IMAGENET_MEAN,
                   std=IMAGENET_STD),
    augm.RandomRotation(degrees=(-10.0, 10.0)),
    augm.RandomHorizontalFlip(p=0.5),
    augm.RandomBrightness(brightness=(0.8, 1.2)),
    data_format="BTCHW",
    same_on_frame=True
)

val_transform = augm.VideoSequential(
    augm.Normalize(mean=IMAGENET_MEAN,
                   std=IMAGENET_STD),
    data_format="BTCHW",
    same_on_frame=True
)
