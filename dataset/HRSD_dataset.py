import json
import os
import pickle
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

        to_tensor = transforms.ToTensor()
        image = to_tensor(image_np)
        dav2_depth = to_tensor(dav2_depth)
        depth_pro_depth = to_tensor(depth_pro_depth)
        return image, dav2_depth, depth_pro_depth


def transfer_raw_to_png(raw_path):
    raw = rawpy.imread(raw_path)
    rgb = raw.postprocess()
    save_path = raw_path.replace('.raw', '.png')
    cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def get_meta(meta_json):
    json_list = []
    with open(meta_json, 'r') as f:
        for line in f:
            data = json.loads(line)
            image_path = data['img_path']
            id = data['id']
            dav2_path = os.path.join(data_root, 'AM2K-dav2', f'{id}.png')
            depth_pro_path = os.path.join(data_root, 'AM2K-depthpro', f'{id}.png')
            json_list.append({
                'id': id,
                'img_path': image_path,
                'dav2_path': dav2_path,
                'depth_pro_path': depth_pro_path,
            })

    with open(meta_json, 'w') as f:
        for entry in json_list:
            json.dump(entry, f)
            f.write('\n')


if __name__ == "__main__":

    if not os.path.exists(meta_json):
        get_meta(meta_json=meta_json)

    dataset = HRSD_Dataset()
    print(f"Dataset length: {len(dataset)}")

    os.makedirs('./vis/syn_am2k', exist_ok=True)
    for id, data in enumerate(dataset):
        image, dav2_depth, depth_pro_depth = data

        save_image(image, f'./vis/syn_am2k/{id}_image.png')
        save_image(dav2_depth, f'./vis/syn_am2k/{id}_dav2_depth.png')
        save_image(depth_pro_depth, f'./vis/syn_am2k/{id}_depth_pro_depth.png')
