import os
import pandas as pd
import numpy as np
import torch
import argparse


def read_single_csv(csv_img_path):
    df = pd.read_csv(csv_img_path, skipinitialspace=True, na_filter=False)
    if '#x' in df.keys():  # fix some Vaa3d garbage
        df.rename(columns={'#x': 'x'}, inplace=True)
    if '##x' in df.keys():  # fix some Vaa3d garbage
        df.rename(columns={'##x': 'x'}, inplace=True)

    df.rename(columns={'x': 'z', 'z': 'x'}, inplace=True)
    centers_df = df.iloc[:, :3]
    centers = [[center.x, center.y, center.z] for _,center in centers_df.iterrows()]

    return centers


def contains_centers(coordinates, centers, patch_size,
                     min_centers=1, padding=5):
    coordinates = np.array(coordinates)
    n_centers_contained = 0
    for center in centers:
        contained = (((coordinates + padding) < center).all() and
                     (((coordinates - padding) + patch_size) > center).all())
        if contained:
            n_centers_contained += 1
            if n_centers_contained >= min_centers:
                return True
    return False


def not_too_dark(coordinates, patch_size, original_image, threshold):

    average_gray_level = (np.mean(
        original_image[coordinates[0]:coordinates[0]
                       + patch_size, coordinates[1]:coordinates[1]
                       + patch_size, coordinates[2]:coordinates[2]
                       + patch_size]))

    if average_gray_level > threshold:
        return True
    else:
        return False


counter_graylevel = 0


def patch_generator(img_name, patch_size, img_dir, csv_dir, th=8,
                    perc=0.25, n_centers=1):
    '''extract patches from an entire substack
    without an overlap of at most 50%'''
    counter_graylevel= 0
    overlap_percentage = (patch_size**3) * perc

    patch_distance =  int(patch_size * perc)


    img_path = os.path.join(img_dir, img_name) + ".pth"
    csv_img_path = os.path.join(csv_dir, img_name) + "-GT.marker"
    torch_image = torch.load(img_path)
    original_image = torch_image.numpy()

    mask = np.zeros(np.array(original_image.shape) - patch_size)

    coord_list = []

    centers = read_single_csv(csv_img_path)

    xs, ys, zs = np.where(mask == 0)
    choosen_points = [[x,y,z] for x,y,z in zip(xs,ys,zs) if (x % patch_distance == 0 and y % patch_distance == 0 and z % patch_distance == 0)]
    #points = sorted(zip(xs, ys, zs))
    np.random.shuffle(choosen_points)
    #print('choosen points')
    #print(choosen_points)

    for x, y, z in choosen_points:  # lento provare tutti i punti, possibilta' di
                            # introdurre un'altra condiione di arresto
        '''
        patch_mask = mask[x:x+patch_size,
                          y:y+patch_size,
                          z:z+patch_size]
        overlap_area = np.sum(patch_mask)
        '''
        #print([x,y,z])
        contains_cells = contains_centers([x, y, z],
                                          centers, patch_size, n_centers)
        acceptable_graylevel = not_too_dark([x, y, z],
                                            patch_size, original_image, th)
        counter_graylevel += 1 if acceptable_graylevel and not contains_cells  else 0
        if  contains_cells or acceptable_graylevel:
            #print("choosen : "+ str([x,y,z]))
            mask[x:x+patch_size, y:y+patch_size, z:z+patch_size] = 1

            coord_list.append([x, y, z, img_name])
    #print("coord_list: ")
    #print(coord_list)
    return pd.DataFrame(coord_list, columns=['x', 'y', 'z', 'img_name']), counter_graylevel


def patch_balancer(image_dir, csv_dir,target_dir, patch_size):
    '''Starting from the whole set of substacks it creates a dataset of
    patch in a balanced fashion'''

    img_names = [name.split(".")[0] for name in os.listdir(image_dir)  if name.split(".")[-1] == 'pth']
    print(img_names)
    df = pd.DataFrame(columns=['x', 'y', 'z', 'img_name'])
    counter_graylevel =0
    for img_name in img_names:  # XXX in lettura pd scrivere ignore_index=True
        print img_name
        res = patch_generator(img_name, patch_size, image_dir, csv_dir)

        #
        df=df.append(
            res[0], ignore_index=True)

        counter_graylevel += res[1]

    path_save_csv = os.path.join(target_dir, "patches.csv")


    file = open(path_save_csv, 'a')
    comment = '# size='+ str(patch_size) + '\n'
    file.write(comment)
    df.to_csv(file)
    file.close()

    print counter_graylevel

def get_parser():
    parser = argparse.ArgumentParser(
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_dir', dest = 'image_dir', type=str,
                        help="""Directory containing 3d images""")

    parser.add_argument('--csv_dir', dest='csv_dir', type=str,
                        help="""Directory containing csv ground truth """)

    parser.add_argument('--target_dir', dest='target_dir', type=str,
                        help="""Directory to put the patches.csv file in """)

    parser.add_argument('--patch_size', dest='patch_size', type=int,
                        help="""Patch dimension """)
    return parser

if __name__ == "__main__":

    parser = get_parser()

    args = parser.parse_args()
    image_dir = args.image_dir
    csv_dir = args.csv_dir
    target_dir = args.target_dir
    patch_size = args.patch_size

    patch_balancer(image_dir,csv_dir, target_dir, patch_size = patch_size)
