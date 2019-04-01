import os
import pickle
from torch.utils.data import Dataset
import torch
import pandas as pd
# from PIL import Image
# from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import tifffile

class DataReader(Dataset):
    """Documentation for DataReader
    Use a csv to select patch from entire substacks
    both from original and gt ones
    """
    def __init__(self, img_dir, gt_dir, df, patch_size, transform=None):
        super(DataReader, self).__init__()
        self.img_dir = img_dir
        self.gt_dir = gt_dir

        self.patch_df = df
        self.transforms = transform
        self.patch_size = patch_size

    def __len__(self):
        return self.patch_df.shape[0]

    def __getitem__(self, idx):
        x, y, z, img_name = self.patch_df.iloc[idx]
        # img_name = str(img_name)
        img_path = os.path.join(self.img_dir, img_name) + ".pth"
        # img_path = os.path.join(self.img_dir, img_name) + "-GT.pth"
        gt_path = os.path.join(self.gt_dir, img_name) + "-GT.pth"
        # print img_path
        image = torch.load(img_path)
        gt = torch.load(gt_path)

        original_patch = image[x:x + self.patch_size,
                               y:y + self.patch_size,
                               z:z + self.patch_size].float()
        gt_patch = gt[x:x + self.patch_size,
                      y:y + self.patch_size,
                      z:z + self.patch_size].float()

        return original_patch / 255, gt_patch / 255


class DataReaderWeight(Dataset):
    """Documentation for DataReader
    magi
    """
    def __init__(self, img_dir, gt_dir, df, patch_size, transform=None):
        super(DataReaderWeight, self).__init__()
        self.img_dir = img_dir
        self.gt_dir = gt_dir

        self.patch_df = df
        self.transforms = transform
        self.patch_size = patch_size

    def __len__(self):
        return self.patch_df.shape[0]

    def __getitem__(self, idx):
        x, y, z, img_name = self.patch_df.iloc[idx]
        # img_name = str(img_name)
        img_path = os.path.join(self.img_dir, img_name) + ".pth"
        # img_path = os.path.join(self.img_dir, img_name) + "-GT.pth"
        gt_path = os.path.join(self.gt_dir, img_name) + "-GT.pth"
        # print img_path
        image = torch.load(img_path)
        gt = torch.load(gt_path)

        original_patch = image[x:x + self.patch_size,
                               y:y + self.patch_size,
                               z:z + self.patch_size].float()
        gt_patch = gt[x:x + self.patch_size,
                      y:y + self.patch_size,
                      z:z + self.patch_size].float()

        # mask = torch.zeros((self.patch_size, self.patch_size, self.patch_size))
        mask = gt_patch != 0

        return original_patch / 255, gt_patch / 255, mask.float()


import numpy as np
if __name__ == "__main__":
    csv_path = "/home/leonardo/workspaces/bcfind/dataset/patches.csv"
    # csv_path = "/home/cosimo/machine_learning_dataset/patches.csv"
    # orig_path = "/home/leonardo/workspaces/bcfind/dataset/3d_img"
    gt_path = "/home/leonardo/workspaces/bcfind/dataset/GT/3d_gt"
    orig_path = "/home/leonardo/workspaces/bcfind/dataset/3d_img"

    complete_dataframe = pd.read_csv(csv_path, comment="#",
                                     index_col=0, dtype={"img_name": str})
    patch_size = int(str(pd.read_csv(csv_path, nrows=1, header=None).
                         take([0])).split("=")[-1])  # workaround

    data_reader = DataReader_weight(orig_path, gt_path,
                                    complete_dataframe, patch_size)
    data_loader = DataLoader(data_reader, 1,
                             shuffle=True, num_workers=1)

    # train, gt, mask = data_reader[0]
    # print train

    total_iteration = len(data_loader)
    progress_bar = tqdm(enumerate(data_loader), total=total_iteration)
    for i, patches in progress_bar:
        img_patches, gt_patches, mask = patches
        name = str(np.random.randint(1000))
        gt_save = (gt_patches[0] * 255).numpy().astype(np.uint8)
        img_save = (patches[0] * 255).numpy().astype(np.uint8)
        mask_save = (mask[0] * 255).numpy().astype(np.uint8)
        print np.unique(gt_save - img_save)
        #print i
        tifffile.imwrite('/home/leonardo/Desktop/tiff_test/' + name + '.tif',
                         img_save, photometric='minisblack')

        tifffile.imwrite('/home/leonardo/Desktop/tiff_test/' + name + '_mask.tif',
                         mask_save, photometric='minisblack')

        tifffile.imwrite('/home/leonardo/Desktop/tiff_test/'
                         + name + '_gt.tif',
                         gt_save, photometric='minisblack')

        # train, gt = data_reader[0]
    # print train
