import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.BaseDataset import BaseDataset
from dataset.utils import get_hdf5_array
import cv2
from torchvision.utils import save_image
import rawpy
from joblib import Parallel, delayed

raw_meta_json = '/dataset/sharedir/research/HRSD/MonoDepth_HRSD_GTA/raw_meta_data.json'
meta_json = '/dataset/sharedir/research/HRSD/MonoDepth_HRSD_GTA/meta_data.json'
data_root = '/dataset/sharedir/research/HRSD/MonoDepth_HRSD_GTA/GTA/'


class HRSD_Dataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.meta_json = meta_json
        self.image_paths = []
        self.depth_paths = []
        with open(meta_json, "r", encoding="utf-8") as infile:
            for line in infile:
                entry = json.loads(line)
                self.image_paths.append(entry["img_path"])
                self.depth_paths.append(entry["depth_path"])

    def __len__(self):
        return len(self.image_paths)

    def preproess_img(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0

        # depth = np.clip(depth, 0.0, self.depth_threshold)

        return image

    def preprocess_depth(self, depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth = depth.astype(np.float32)
        depth = depth / 255.0
        return depth

    def __getitem__(self, idx):
        '''
        idx: list,int
        Return:
            image: torch.Tensor
            depth: torch.Tensor
        '''
        if isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]
        image_np = self.preproess_img(self.image_paths[idx])
        depth_np = self.preprocess_depth(self.depth_paths[idx])
        to_tensor = transforms.ToTensor()
        image = to_tensor(image_np)
        depth = to_tensor(depth_np)
        return image, depth


def convertColorRAW2PNG(filepath, dsize=(720, 1280)):
    """
    Convert RAW file to PNG using custom buffer processing.

    Args:
        filepath (str): Path to the .raw file.
        dsize (tuple): Dimensions (height, width) of the image.

    Returns:
        str: Path to the saved PNG file.
    """
    print(f"Processing {filepath}")
    save_path = os.path.join(
        os.path.split(filepath)[0],
        os.path.split(filepath)[1].split('-')[0] + "Color.png"
    )

    if not os.path.exists(save_path) or cv2.imread(save_path) is None:
        with open(filepath, 'rb') as file:
            colorBuf = file.read()
        color = np.frombuffer(colorBuf, dtype=np.uint8)
        try:
            image = np.reshape(color, (dsize[0], dsize[1], 4), 'C')
        except ValueError:
            print(f"Bad color file: {filepath}")
            return None

        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return save_path


def convertDepthRAW2PNG(filepath, dsize=(720, 1280)):
    print(f"Processing {filepath}")
    Relative_save_path = os.path.join(
        os.path.split(filepath)[0],
        os.path.split(filepath)[1].split('-')[0] + "Depth.png"
    )
    if not os.path.exists(Relative_save_path) or cv2.imread(Relative_save_path) is None:
        with open(filepath, 'rb') as file:
            depthBuf = file.read()
        depth = np.frombuffer(depthBuf, dtype=np.float32)
        try:
            image = np.reshape(depth, dsize, 'C')
        except ValueError:
            print(f"Bad depth file: {filepath}")
            return
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = (image * 255).astype(np.uint8)

        cv2.imwrite(Relative_save_path, image)


def get_meta(meta_json, data_root, dsize=(720, 1280), n_jobs=-1):
    """
    Process all .raw files in the given directory and convert them to PNG.

    Args:
        meta_json (str): Metadata file (unused in this implementation).
        data_root (str): Root directory containing .raw files.
        dsize (tuple): Dimensions (height, width) of the images.
        n_jobs (int): Number of parallel jobs (-1 for all available CPUs).

    Returns:
        list: List of paths to the converted PNG files.
    """
    raw_img_files = []
    raw_depth_files = []
    if not os.path.exists(raw_meta_json):

        # Collect all .raw files
        for root, _, files in os.walk(data_root):
            for file in files:
                if file.endswith('.raw'):
                    file_path = os.path.join(root, file)
                    if '-color.raw' in file_path:
                        raw_img_files.append(file_path)
        raw_depth_files = [file.replace('-color.raw', '-depth.raw') for file in raw_img_files]
        with open(raw_meta_json, 'w') as f:
            for id, (img_path, depth_path) in enumerate(zip(raw_img_files, raw_depth_files)):
                json.dump({
                    'id': id,
                    'img_path': img_path,
                    'depth_path': depth_path
                }, f)
                f.write('\n')

    with open(raw_meta_json, 'r') as f:
        for line in f:
            entry = json.loads(line)
            raw_img_files.append(entry["img_path"])
            raw_depth_files.append(entry["depth_path"])
    print(f"Total images: {len(raw_img_files)} | Total depth: {len(raw_depth_files)}")
    print(f"Raw image files: {raw_img_files[:5]}")
    print(f"Raw depth files: {raw_depth_files[:5]}")
    # input("Press Enter to continue...")
    # Use joblib for parallel processing
    Parallel(n_jobs=n_jobs)(
        delayed(convertColorRAW2PNG)(file_path, dsize) for file_path in raw_img_files
    )
    Parallel(n_jobs=n_jobs)(
        delayed(convertDepthRAW2PNG)(file_path, dsize) for file_path in raw_depth_files
    )

    png_img = []
    png_depth = []
    # Filter out None values
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                if 'color' in file_path.lower():
                    depth_path = file_path.replace('Color', 'Depth')
                    if os.path.exists(depth_path):
                        if cv2.imread(file_path) is not None and cv2.imread(depth_path) is not None:
                            png_img.append(file_path)
                            png_depth.append(depth_path)

    with open(meta_json, 'w') as f:

        for id, (image_path, depth_path) in enumerate(zip(png_img, png_depth)):
            type = 'train'
            if 'val' in image_path:
                type = 'validation'
            json.dump({
                'id': id,
                'type': type,
                'img_path': image_path,
                'depth_path': image_path.replace('-color', '-depth')
            }, f)
            f.write('\n')


def check_data(index, data):
    """
    Function to validate a single dataset entry.

    Args:
        index (int): Index of the data.
        data (tuple): A tuple containing the image and depth data.

    Returns:
        str: Validation result as a formatted string.
    """
    image, depth = data
    print(f"ID: {index} | Image shape: {image.shape} | Depth shape: {depth.shape}")


def validate_dataset(dataset, n_jobs=-1):
    """
    Validate the dataset using multiple threads.

    Args:
        dataset (iterable): The dataset containing (image, depth) pairs.
        n_jobs (int): Number of threads to use (-1 for all available).

    Returns:
        None
    """
    Parallel(n_jobs=n_jobs)(
        delayed(check_data)(idx, data) for idx, data in enumerate(dataset)
    )


if __name__ == "__main__":

    if not os.path.exists(meta_json):
        get_meta(meta_json=meta_json, data_root=data_root)

    dataset = HRSD_Dataset()
    print(f"Dataset length: {len(dataset)}")
    validate_dataset(dataset)
