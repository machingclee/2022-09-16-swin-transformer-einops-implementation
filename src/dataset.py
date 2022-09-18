from cProfile import label
from matplotlib import image
import numpy as np
import torch
import albumentations as A
import json
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from . import config
from torchvision import transforms
from torchvision.transforms import ToPILImage
from copy import deepcopy
from typing import List, TypedDict


to_tensor = transforms.ToTensor()

torch_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def resize_img(img):
    """
    img:  Pillow image
    """
    h, w = img.height, img.width
    if h >= w:
        ratio = config.input_height / h
        new_h, new_w = int(h * ratio), int(w * ratio)
    else:
        ratio = config.input_width / w
        new_h, new_w = int(h * ratio), int(w * ratio)

    img = img.resize((new_w, new_h), Image.BILINEAR)
    return img, (w, h)


def pad_img(img):
    h = img.height
    w = img.width
    img = np.array(img)
    img = np.pad(img, pad_width=((0, config.input_height - h), (0, config.input_width - w), (0, 0)), mode="constant")
    img = Image.fromarray(img)
    assert img.height == config.input_height
    assert img.width == config.input_width
    return img


def resize_and_padding(img, return_window=False):
    img, (ori_w, ori_h) = resize_img(img)
    w = img.width
    h = img.height
    padding_window = (w, h)
    img = pad_img(img)

    if not return_window:
        return img
    else:
        return img, padding_window, (ori_w, ori_h)


albumentation_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0, rotate_limit=10, p=0.7),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(p=0.5)
    ], p=0.8),

    A.LongestMaxSize(max_size=config.input_height, interpolation=1, p=1),
    A.PadIfNeeded(
        min_height=config.input_height,
        min_width=config.input_height,
        border_mode=0,
        value=(0, 0, 0),
        position="top_left"
    ),
],
    p=1,
    bbox_params=A.BboxParams(format="pascal_voc", min_area=0.1)
)


class Target(TypedDict):
    image_path: str
    boxes: List[List[float]]
    segmentations: List[List[float]]


class AnnotationDataset(Dataset):
    def __init__(self, mode="train"):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = AnnotationDataset()
    img, boxes_, cls_idxes, mask_pooling_pred_targets = dataset[0]
    print(img)
    print(boxes_)
    print(cls_idxes)
    print(mask_pooling_pred_targets)
