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

meta_json = '/dataset/sharedir/research/HRSD/MonoDepth_HRSD_GTA/GTA/meta_data.json'
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
    save_path = None
    if not os.path.exists(filepath):
        with open(filepath, 'rb') as file:
            colorBuf = file.read()
        color = np.frombuffer(colorBuf, dtype=np.uint8)
        image = np.reshape(color, (dsize[0], dsize[1], 4), 'C')
        save_path = os.path.join(
            os.path.split(filepath)[0],
            os.path.split(filepath)[1].split('-')[0] + "Color.png"
        )
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return save_path


def convertDepthRAW2PNG(filepath, dsize=(720, 1280)):
    print(f"Processing {filepath}")
    with open(filepath, 'rb') as file:
        depthBuf = file.read()
    depth = np.frombuffer(depthBuf, dtype=np.float32)
    image = np.reshape(depth, dsize, 'C')
    f = 10003.814
    n = 0.15
    numerator = (-f * n)
    denominator = (((n - f) * image) - n)
    cv2.imwrite(os.path.join(os.path.split(filepath)[0], os.path.split(filepath)[1].split('-')[0] + "DepthReal.pfm"),
                numerator / denominator)
    cv2.imwrite(
        os.path.join(os.path.split(filepath)[0], os.path.split(filepath)[1].split('-')[0] + "DepthRelative.pfm"), image)


def process_img(file_path, dsize=(720, 1280)):
    """
    Process a single .raw file to convert it to PNG.

    Args:
        file_path (str): Path to the .raw file.
        dsize (tuple): Dimensions (height, width) of the image.

    Returns:
        str or None: Path to the saved PNG file if '-color' in name, else None.
    """
    save_path = convertColorRAW2PNG(file_path, dsize)
    if '-color' in save_path:
        return save_path
    return None


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
    # Collect all .raw files
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith('.raw'):
                if 'color.raw' in file:
                    raw_img_files.append(os.path.join(root, file))
                elif 'depth.raw' in file:
                    raw_depth_files.append(os.path.join(root, file))
    print(f"Total images: {len(raw_img_files)} | Total depth: {len(raw_depth_files)}")
    # Use joblib for parallel processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_img)(file_path, dsize) for file_path in raw_img_files
    )
    Parallel(n_jobs=n_jobs)(
        delayed(convertDepthRAW2PNG)(file_path, dsize) for file_path in raw_depth_files
    )

    # Filter out None values
    image_list = [result for result in results if result is not None]

    with open(meta_json, 'w') as f:
        cnt = 0
        for image_path in image_list:
            cnt += 1
            type = 'train'
            if 'val' in image_path:
                type = 'validation'
            json.dump({
                'id': cnt,
                'type': type,
                'img_path': image_path,
                'depth_path': image_path.replace('-color', '-depth')
            }, f)
            f.write('\n')


if __name__ == "__main__":

    if not os.path.exists(meta_json):
        get_meta(meta_json=meta_json, data_root=data_root)

    dataset = HRSD_Dataset()
    print(f"Dataset length: {len(dataset)}")

    os.makedirs('./vis/syn_am2k', exist_ok=True)
    for id, data in enumerate(dataset):
        image, dav2_depth, depth_pro_depth = data

        save_image(image, f'./vis/syn_am2k/{id}_image.png')
        save_image(dav2_depth, f'./vis/syn_am2k/{id}_dav2_depth.png')
        save_image(depth_pro_depth, f'./vis/syn_am2k/{id}_depth_pro_depth.png')
