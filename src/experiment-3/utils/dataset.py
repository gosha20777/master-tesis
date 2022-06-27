from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import imageio
import PIL.Image as Image
import torch
import os
import utils.image as image

from utils.utils import extract_bayer_channels

to_tensor = transforms.Compose([transforms.ToTensor()])

class LoadData(Dataset):
    """Helper to iterate over the data (as Numpy arrays)."""
    def __init__(self,
            img_size, 
            dslr_scale, 
            input_img_paths, 
            target_img_paths
        ):
        self.img_size = img_size
        self.dslr_scale = dslr_scale
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths)

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        dslr_size = (self.img_size[0] * self.dslr_scale, self.img_size[1] * self.dslr_scale)
        
        raw_image = image.read_bayer_image(self.input_img_paths[idx])
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))
        x = raw_image

        img = image.read_target_image(self.target_img_paths[idx], dslr_size)
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        y = img
        
        return x, y, str(idx)