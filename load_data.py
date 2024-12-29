import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class StereoDataset(Dataset):
    def __init__(self, left_images, right_images, disparity_maps):
        # Normalize images to [0, 1] range
        self.left_images = torch.tensor(left_images / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
        self.right_images = torch.tensor(right_images / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
        # Add channel dimension to disparity maps and normalize
        self.disparity_maps = torch.tensor(disparity_maps, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        return self.left_images[idx], self.right_images[idx], self.disparity_maps[idx]


def load_kitti_data(dataset_root):
    data = {"left_images": [], "right_images": [], "disparity": []}
    left_dir = os.path.join(dataset_root, 'image_2')
    right_dir = os.path.join(dataset_root, 'image_3')
    disp_dir = os.path.join(dataset_root, 'disp_occ_0')

    for filename in sorted(os.listdir(left_dir)):
        if filename.endswith("10.png"):
            left_path = os.path.join(left_dir, filename)
            right_path = os.path.join(right_dir, filename.replace('10.png', '11.png'))
            disp_path = os.path.join(disp_dir, filename)

            if os.path.exists(left_path) and os.path.exists(right_path) and os.path.exists(disp_path):
                left_image = cv2.imread(left_path)
                right_image = cv2.imread(right_path)
                disparity = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0  # Convert to float

                if left_image is not None and right_image is not None and disparity is not None:
                    data["left_images"].append(left_image)
                    data["right_images"].append(right_image)
                    data["disparity"].append(disparity)

    return data