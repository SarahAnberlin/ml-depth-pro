from torchvision.utils import save_image
# import interpolation mode
from torchvision.transforms import InterpolationMode
import json
import os
import time
import cv2
import numpy as np
import torchvision.transforms
from PIL import Image
from matplotlib import pyplot as plt
import torch
import src.depth_pro.depth_pro as depth_pro
from dataset.AM2KDataset import AM2KDataset
from dataset.HypersimDataset import HypersimDataset
from dataset.NYUDataset import NYUDataset
from dataset.SintelDataset import SintelDataset
from dataset.utils import get_hdf5_array
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model = model.to(device).eval()


def save_single_fig(predict_depth, save_root, id):
    # Normalize to range [0, 1]
    predict_depth = (predict_depth - np.min(predict_depth)) / (np.max(predict_depth) - np.min(predict_depth))

    # Apply colormap
    cmap = plt.get_cmap('viridis')  # 使用 'viridis' 颜色映射，可更改为其他映射
    predict_depth_colored = cmap(predict_depth)  # 返回 RGBA 数组
    predict_depth_colored = (predict_depth_colored[:, :, :3] * 255).astype(np.uint8)  # 转换为 RGB

    # Save as PNG using PIL
    save_path = os.path.join(save_root, f"{id}.png")
    print(f"Saving to {save_path}")
    image = Image.fromarray(predict_depth_colored)
    image.save(save_path)


def clip_array_percentile(array, lower_percentile=20, upper_percentile=80):
    """
    Clips the values in the NumPy array to the range defined by the lower and upper percentiles.

    Parameters:
        array (np.ndarray): Input array.
        lower_percentile (float): The lower percentile (default: 20).
        upper_percentile (float): The upper percentile (default: 80).

    Returns:
        np.ndarray: The clipped array.
    """
    # Calculate the percentile values
    lower_bound = np.percentile(array, lower_percentile)
    upper_bound = np.percentile(array, upper_percentile)

    # Clip the array
    clipped_array = np.clip(array, lower_bound, upper_bound)

    return clipped_array


def get_dataset(dataset_name):
    if dataset_name == "Hypersim":
        return HypersimDataset()
    elif dataset_name == "Sintel":
        return SintelDataset()
    elif dataset_name == "NYUv2":
        return NYUDataset()
    elif dataset_name == 'AM2K':
        return AM2KDataset()


# dataset_name = "Sintel"
# dataset_name = "Hypersim"
dataset_name = "AM2K"
dataset = get_dataset(dataset_name)
save_root = os.path.join('./vis/depth-pro-test-large', dataset_name)
os.makedirs(save_root, exist_ok=True)
cnt = 0
elapse_time = 0.0
for id, data in enumerate(dataset):
    if id % 10 == 0:
        print(f"Processing {id + 1}/{len(dataset)}")
    image = data
    # depth = data[1]
    image = image.unsqueeze(0).to(device)
    h, w = image.shape[-2:]
    image = torchvision.transforms.Resize((1536, 1536), interpolation=InterpolationMode.BICUBIC)(image)

    # Run inference.
    with torch.no_grad():
        prediction, _ = model(image * 2 - 1, test=True)
    prediction = torchvision.transforms.Resize((h, w), interpolation=InterpolationMode.BILINEAR)(prediction)
    depth = prediction
    depth = depth.squeeze()
    save_image(depth, os.path.join(save_root, f"{id + 1}.png"))
