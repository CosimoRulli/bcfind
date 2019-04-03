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
from bcfind import volume
import bcfind.volume
import argparse
from bcfind import mscd


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
    def __init__(self, img_dir, gt_dir, weight_dir, df,
                 patch_size, transform=None):
        super(DataReaderWeight, self).__init__()
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.weight_dir = weight_dir

        self.patch_df = df
        self.transforms = transform
        self.patch_size = patch_size

    def __len__(self):
        return self.patch_df.shape[0]

    def __getitem__(self, idx):
        x, y, z, img_name = self.patch_df.iloc[idx]

        img_path = os.path.join(self.img_dir, img_name) + ".pth"
        gt_path = os.path.join(self.gt_dir, img_name) + "-GT.pth"
        weight_path = (os.path.join(self.weight_dir, img_name)
                       + "_weighted_map.pth")

        image = torch.load(img_path)
        gt = torch.load(gt_path)
        weighted_map = torch.load(weight_path)

        original_patch = image[x:x + self.patch_size,
                               y:y + self.patch_size,
                               z:z + self.patch_size].float()
        gt_patch = gt[x:x + self.patch_size,
                      y:y + self.patch_size,
                      z:z + self.patch_size].float()

        weight_patch = weighted_map[x:x + self.patch_size,
                                    y:y + self.patch_size,
                                    z:z + self.patch_size].float()

        return original_patch / 255, gt_patch / 255, weight_patch


class DataReaderWeight_old(Dataset):
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

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('csv_path', metavar='csv_path', type=str,
                        help="""path for csv patches file""")

    parser.add_argument('img_dir', metavar='img_dir', type=str,
                        help="""Directory contaning the collection
                            of pth images""")

    parser.add_argument('gt_dir', metavar='gt_dir', type=str,
                        help="""Directory contaning the collection
                            of pth gt images""")

    parser.add_argument('weight_dir', metavar='weight_dir', type=str,
                        help="""Directory contaning the collection
                            of weighted_map""")

    parser.add_argument('outdir', metavar='outdir', type=str,
                        help="""Directory where prediction results will be saved, e.g. outdir/100905/ms.marker.
                        Will be created or overwritten""")

    parser.add_argument('-r', '--hi_local_max_radius', metavar='r', dest='hi_local_max_radius',
                        action='store', type=float, default=6,
                        help='Radius of the seed selection ball (r)')



    parser.add_argument('-f', '--floating_point', dest='floating_point', action='store_true',
                        help='If true, cell centers are saved in floating point.')

    parser.add_argument('-m', '--min_second_threshold', metavar='min_second_threshold', dest='min_second_threshold',
                        action='store', type=int, default=15,
                        help="""If the foreground (second threshold in multi-Kapur) is below this value
                           then the substack is too dark and assumed to contain no soma""")
    parser.add_argument('-l', '--local', dest='local', action='store_true',
                        help='Perform local processing by dividing the volume in 8 parts.')
    # parser.set_defaults(local=True)
    parser.add_argument('-t', '--seeds_filtering_mode', dest='seeds_filtering_mode',
                        action='store', type=str, default='soft',
                        help="Type of seed selection ball ('hard' or 'soft')")
    parser.add_argument('-R', '--mean_shift_bandwidth', metavar='R', dest='mean_shift_bandwidth',
                        action='store', type=float, default=5.5,
                        help='Radius of the mean shift kernel (R)')
    parser.add_argument('-s', '--save_image', dest='save_image', action='store_true',
                        help='Save debugging substack for visual inspection (voxels above threshold and colorized clusters).')
    parser.add_argument('-M', '--max_expected_cells', metavar='max_expected_cells', dest='max_expected_cells',
                        action='store', type=int, default=10000,
                        help="""Max number of cells that may appear in a substack""")
    parser.add_argument('-p', '--pair_id', dest='pair_id',
                        action='store', type=str,
                        help="id of the pair of views, e.g 000_090. A folder with this name will be created inside outdir/substack_id")
    parser.set_defaults(save_image=False)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    #csv_path = "/home/leonardo/workspaces/bcfind/dataset/patches.csv"
    #csv_path = "/home/cosimo/machine_learning_dataset/patches.csv"
    # orig_path = "/home/leonardo/workspaces/bcfind/dataset/3d_img"

    #gt_path = "/home/leonardo/workspaces/bcfind/dataset/GT/3d_gt"
    #gt_path = "/home/cosimo/machine_learning_dataset/gt_images_w_mask"
    #weight_path = "/home/cosimo/machine_learning_dataset/gt_images_w_mask"

    #orig_path = "/home/leonardo/workspaces/bcfind/dataset/3d_img"
    #orig_path = "/home/cosimo/machine_learning_dataset/3DImages_noflip"
    csv_path = args.csv_path
    orig_path= args.img_dir
    gt_path = args.gt_dir
    weight_path = args.weight_dir

    complete_dataframe = pd.read_csv(csv_path, comment="#",
                                     index_col=0, dtype={"img_name": str})
    patch_size = int(str(pd.read_csv(csv_path, nrows=1, header=None).
                         take([0])).split("=")[-1])

    data_reader = DataReaderWeight(orig_path, gt_path, weight_path,
                                    complete_dataframe, patch_size)
    data_loader = DataLoader(data_reader, 1,
                             shuffle=True, num_workers=1)

    # train, gt, mask = data_reader[0]
    # print train

    total_iteration = len(data_loader)
    progress_bar = tqdm(enumerate(data_loader), total=total_iteration)
    for i, patches in progress_bar:
        img_patches, gt_patches, mask = patches


        img_patch_numpy = img_patches[0].numpy()
        substack_id = '0'
        substack_dict = {substack_id: {'Files': 'dummy files', 'Height': 64, 'Width': 64, 'Depth': 64}}
        plist = {'Height': 64, 'Width': 64, 'Depth': 64, 'SubStacks': substack_dict}
        patch = volume.SubStack('', substack_id, plist)
        patch.load_volume_from_3D(img_patch_numpy)
        mscd.ms(patch, args )
        #centers = gt_patches == 255

        print(centers)
        break
        #name = str(np.random.randint(1000))
        #gt_save = (gt_patches[0] * 255).numpy().astype(np.uint8)
        #img_save = (patches[0] * 255).numpy().astype(np.uint8)
        #mask_save = (mask[0] * 255).numpy().astype(np.uint8)
        #print np.unique(gt_save - img_save)
        #print i

        '''
        tifffile.imwrite('/home/leonardo/Desktop/tiff_test/' + name + '.tif',
                         img_save, photometric='minisblack')

        tifffile.imwrite('/home/leonardo/Desktop/tiff_test/' + name + '_mask.tif',
                         mask_save, photometric='minisblack')

        tifffile.imwrite('/home/leonardo/Desktop/tiff_test/'
                         + name + '_gt.tif',
                         gt_save, photometric='minisblack')

        # train, gt = data_reader[0]
        '''
    # print train
