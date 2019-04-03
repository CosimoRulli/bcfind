import numpy as np
import pandas as pd
import os
from skimage.filters import gaussian
from scipy.ndimage.filters import gaussian_filter
import tifffile
import argparse
from skimage.morphology import binary_dilation
from skimage.morphology import ball
import pickle
import torch

dim = (280, 245, 281)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    parser.add_argument('root_dir', metavar='root_dir', type=str,
                        help="""Directory contaning csv of soma centers""")

    parser.add_argument('target_dir', metavar='targer_dir', type=str,
                        help="""Directory where the gt images
                        will be stored""")

    parser.add_argument('target_dir_mask', metavar='targer_dir_mask', type=str,
                        help="""Directory where the weighted mask
                            will be stored""")

    parser.add_argument('sigma', metavar='sigma', type=float,
                        default=3.5,
                        help="""Sigma of the gaussian filter""")

    parser.add_argument('truncate', metavar='truncate', type=float,
                        default=1.5,
                        help="""parameter for truncate gaussin
                        filter wrt sigma""")
    parser.add_argument('--visualize', dest='visualize', action="store_true",
                        help="""Creates also the tif img""")

    parser.add_argument("--flip", dest ="flip", action="store_true",
                        help = "flip y axis")

    parser.set_defaults(flip = False)
    parser.set_defaults(visualize = False)
    return parser


def place_centers(mask, center_df):
    for idx, center in center_df.iterrows():
        mask[center.x, center.y, center.z] = 255
    return mask


def place_centers_dilated(mask, center_df):
    temp = np.zeros(dim)
    for idx, center in center_df.iterrows():
        temp[center.x, center.y, center.z] = 1
    temp = binary_dilation(temp, ball(5))
    mask[temp] = 255
    return mask


def gt_builder_from_gaussian_filter(csv_path, sigma, truncate, flip):

    df = pd.read_csv(csv_path, skipinitialspace=True, na_filter=False)
    if '#x' in df.keys():  # fix some Vaa3d garbage
        df.rename(columns={'#x': 'x'}, inplace=True)
    if '##x' in df.keys():  # fix some Vaa3d garbage
        df.rename(columns={'##x': 'x'}, inplace=True)
    if flip:
        df.rename(columns={'x': 'z', 'z': 'x'}, inplace=True)

    centers_df = df.iloc[:, :3]
    gt_clear = np.zeros(dim, dtype='uint16')

    gt_seed = place_centers_dilated(np.copy(gt_clear), centers_df)

    # gt_img = gaussian(gt_seed, sigma=sigma, truncate=truncate)
    gt_img = gaussian_filter(gt_seed, sigma=sigma, truncate=truncate)
    return gt_img, gt_seed


def create_ground_truth(root_dir, target_dir, target_dir_w_m, sigma=3.5, truncate=1.5,
                        flip=True,
                        visualize=False):
    #flip = False
    print flip
    for _csv in os.listdir(root_dir):
        gt_img, gt_seed = gt_builder_from_gaussian_filter(os.path.join(root_dir, _csv),
                                                 sigma, truncate,
                                                 flip)
        file_name = _csv.split("-")[0]

        print file_name

        torch_gt = torch.from_numpy(gt_img.astype('float16'))
        torch_seed = torch.from_numpy  (gt_seed.astype('float16'))
        # with open(os.path.join(target_dir, file_name + ".pkl"), 'wb') as f:
        #     pickle.dump(torch_gt, f)

        torch.save(torch_gt,
                   os.path.join(target_dir, file_name + "-GT.pth"))
        torch.save(torch_seed,
                   os.path.join(target_dir_w_m, file_name + "_weighted_map.pth"))
        if visualize:
            tifffile.imwrite(os.path.join(target_dir, file_name + ".tif"),
                             gt_img, photometric='minisblack')

            tifffile.imwrite(os.path.join(target_dir, file_name + "_weighted_map.tiff"),
                             gt_seed, photometric = 'minisblack')


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if not os.path.isdir(args.target_dir):
        os.makedirs(args.target_dir)
    if not os.path.isdir(args.target_dir_mask):
        os.makedirs(args.target_dir_mask)

    create_ground_truth(args.root_dir, args.target_dir, args.target_dir_mask,
                        args.sigma, args.truncate, args.flip, args.visualize)

# if __name__ == "__main__":
#     "for testing purpose"
#     path = '/home/leonardo/workspaces/bcfind/dataset/GT/TomoDec13/012011-GT.marker'
#     # path = '/home/leonardo/workspaces/bcfind/dataset/GT/TomoDec13/011312-GT.marker'
#     sigma = 3.5
#     trunc = 1.5
#     res, gt_seed = gt_builder_from_gaussian_filter(path, sigma, trunc)

#     print np.max(np.unique(res))

#     with tifffile.TiffWriter('/home/leonardo/Desktop/Stack.tif') as stack:
#         stack.save(gt_seed)
 #         tifffile.imwrite('/home/leonardo/Desktop/Stack_imwr.tif',
#                          res, photometric='minisblack')
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')

#     # ax.scatter(res[0], res[1], res[2], c='b', marker='o')
#     # # ax.plot_surface(res[0], res[1], res[2], color='b')

#     # # ax.plot(res[0], res[1], res[2], color='b')
#     # plt.show()
