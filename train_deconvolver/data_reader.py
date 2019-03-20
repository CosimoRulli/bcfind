import os
import pickle
from torch.utils.data import Dataset
import torch
# from PIL import Image
# from torchvision import transforms


class DataReaderWIP(Dataset):
    """Documentation for DataReader
    TODO: c'è da mettere un altro path in modo che
    restituisca sia l'immagine di train che il gt
    """
    def __init__(self, root_path, transform=None):
        super(DataReaderWIP, self).__init__()
        self.root_dir = root_path
        self.transforms = transform
        self.img_paths = []
        for _file in os.listdir(self.root_dir):
            self.img_paths.append(
                os.path.join(self.root_dir, _file))
        self.img_paths = list(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        patch_path = self.img_paths[idx]
        with open(patch_path, 'rb') as f:
            patch = pickle.load(f)

        if self.transforms:
            patch = self.transforms(patch)
            # to_tensor = transforms.ToTensor()
            # patch = to_tensor(patch)
        return patch


class DataReader(Dataset):
    """Documentation for DataReader
dummy datareader
    """
    def __init__(self, args):
        super(DataReader, self).__init__()

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        patch = torch.random(13, 13, 13)
        gt_patch = torch.random(13, 13, 13)
        return patch, gt_patch
