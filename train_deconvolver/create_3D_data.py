import numpy as np
import os
#import tifffile
from skimage.external import tifffile as tifffile_sk
import argparse
import torch

def stack_folder_tif(folder_path,folder_name,  new_path, flip):
    file_list = os.listdir(folder_path)
    file_list.sort()
    new_file_path = os.path.join(new_path, folder_name) +'.tif'
    with tifffile_sk.TiffWriter(new_file_path) as stack:
        for filename in file_list:
          np_im = tifffile_sk.imread(os.path.join(folder_path, filename))
          if flip:
            np_im = np.flipud(np_im)
          stack.save(np_im)


def stack_folder_torch(folder_path,folder_name,  new_path, flip):
    file_list = os.listdir(folder_path)
    file_list.sort()
    new_file_path = os.path.join(new_path, folder_name)+'.pth'
    np_images = [tifffile_sk.imread(os.path.join(folder_path, filename)) for filename in file_list]
    if flip:
        np_images = [np.flipud(image) for image in np_images]
    print(len(np_images))
    np_images = np.stack(np_images, axis=0)

    torch_im = torch.from_numpy(np_images)
    torch.save(torch_im, new_file_path)


def create_3D_data(folders_path, new_path, flip):
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    for folder_name in os.listdir(folders_path):
        folder_path = os.path.join(folders_path, folder_name)
        stack_folder_torch(folder_path, folder_name, new_path, flip)
        stack_folder_tif(folder_path, folder_name, new_path, flip)

def get_parser():
    parser = argparse.ArgumentParser(
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images_path', dest = 'folders_path', type=str,
                        help="""Directory contaning substacks' directories""")

    parser.add_argument('--new_folder_dir', dest='new_path', type=str,
                        help="""Directory where the new files (pkl) will be created""")
    parser.add_argument("--flip", dest="flip", action="store_true",
                        help="flip y axis")

    parser.set_defaults(flip=False)


    return parser

if __name__ == "__main__":
    folders_path = "/home/cosimo/machine_learning_dataset/cerebellum-img"
    new_path = "/home/cosimo/machine_learning_dataset/3DImages"
    parser = get_parser()
    args = parser.parse_args()
    print(args.flip)
    create_3D_data(args.folders_path, args.new_path, args.flip)