import os
import pickle
from torch.utils.data import Dataset
import torch
import pandas as pd
# from PIL import Image
# from torchvision import transforms


class DataReader_new(Dataset):
    """Documentation for DataReader
    Use a csv to select patch from entire substacks
    both from original and gt ones
    """
    def __init__(self, img_dir, gt_dir, csv_path, transform=None):
        super(DataReader_new, self).__init__()
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.patch_df = pd.read_csv(csv_path, comment="#",
                                    index_col=0, dtype={"img_name": str})
        self.transforms = transform
        self.patch_size = int(str(pd.read_csv(csv_path, nrows=1, header=None).
                                  take([0])).split("=")[-1])  # workaround
        print self.patch_size

    def __len__(self):
        return self.patch_df.shape[0]

    def __getitem__(self, idx):
        x, y, z, img_name = self.patch_df.iloc[idx]
        img_name = str(img_name)
        img_path = os.path.join(self.img_dir, img_name) + ".pth"
        gt_path = os.path.join(self.gt_dir, img_name) + "-GT.pth"
        print img_path
        image = torch.load(img_path)
        gt = torch.load(gt_path)

        original_patch = image[x:x + self.patch_size,
                               y:y + self.patch_size,
                               z:z + self.patch_size]
        gt_patch = gt[x:x + self.patch_size,
                      y:y + self.patch_size,
                      z:z + self.patch_size]

        return original_patch, gt_patch


class DataReaderWIP(Dataset):

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


if __name__ == "__main__":
    csv_path = "/home/leonardo/Desktop/magi.csv"
    orig_path = "/home/leonardo/workspaces/bcfind/dataset/3d_img"
    gt_path = "/home/leonardo/workspaces/bcfind/dataset/GT/3d_gt"
    data_reader = DataReader_new(orig_path, gt_path, csv_path)
    train, gt = data_reader[0]
    print train