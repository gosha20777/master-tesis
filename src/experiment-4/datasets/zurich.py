import torch.utils.data as data
from utils import preprocess


class ZurichDataset(data.Dataset):
    def __init__(self, 
            img_size, 
            dslr_scale, 
            input_img_paths, 
            target_img_paths
    ) -> None:
        assert len(input_img_paths) == len(target_img_paths), 'invalid image pairs'
        self.img_size = img_size
        self.dslr_scale = dslr_scale
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.input_img_paths)

    def __getitem__(self, index):
        x = preprocess.read_bayer_image(self.input_img_paths[index])
        y = preprocess.read_target_image(self.target_img_paths[index])
        return x, y
