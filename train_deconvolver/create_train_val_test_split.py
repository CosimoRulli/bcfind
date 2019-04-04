

import os
import pandas as pd
import numpy as np
import torch
import argparse
import math
np.random.seed(41)

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
        # counter_graylevel += 1 if acceptable_graylevel and not contains_cells  else 0

        if  contains_cells or acceptable_graylevel:
        #if contains_cells:
            #print("choosen : "+ str([x,y,z]))
            mask[x:x+patch_size, y:y+patch_size, z:z+patch_size] = 1

            coord_list.append([x, y, z, img_name])
    #print("coord_list: ")
    #print(coord_list)
    return pd.DataFrame(coord_list, columns=['x', 'y', 'z', 'img_name'])



def divide_im_names(im_names, test_perc):

    n_images = int(im_names.shape[0] )# 66
    n_test_im = int(math.floor(n_images * test_perc))
    n_train_im = n_images - n_test_im
    print n_test_im
    train_indexes = [i for i in range( n_images)]
    test_indexes = np.random.choice(train_indexes, size = n_test_im, replace=False)

    #test_indexes = np.random.randint(0, n_images, size=n_test_im)
    test_indexes.sort()
    print  test_indexes
    train_indexes = np.array(list(set(train_indexes) - set(test_indexes)))
    print train_indexes
    print len(train_indexes)
    train_images = im_names[train_indexes]
    test_images = im_names[test_indexes]
    return train_images, test_images



def generate_csvs(image_dir, csv_dir, target_dir, train_csv_name, val_test_csv_name,  perc_test, perc_val, patch_size, threshold):

    image_names = os.listdir(image_dir)
    print 'Total images: ' + str(len(image_names))
    image_names = np.array([image.split('.')[0] for image in image_names])

    print 'Train-test split'
    train_images, test_images = divide_im_names(image_names, perc_test)

    print 'Train-val split'
    train_images, val_images = divide_im_names(train_images, perc_val)

    print 'Train images: ' + str(train_images.shape[0])
    print 'Validation images: '+str(len(val_images))
    print 'Test images: ' + str(len(test_images))

    print 'Generating Train csv'
    df_train = pd.DataFrame(columns=['x', 'y', 'z', 'img_name'])
    for train_im in train_images:
        print train_im
        res = patch_generator(train_im, patch_size, image_dir, csv_dir, th=threshold)

        df_train = df_train.append(
            res, ignore_index=True)

    path_save_csv_train = os.path.join(target_dir, train_csv_name + "_patches.csv")

    file = open(path_save_csv_train, 'a')
    comment = '# size=' + str(patch_size) + '\n'
    file.write(comment)
    df_train.to_csv(file)
    file.close()


    df_val_test = pd.DataFrame(columns=['name', 'split'])
    val_csv = [[val_name, 'VAL'] for val_name in val_images]
    df_val = pd.DataFrame(val_csv, columns=['name', 'split'])
    df_val_test = df_val_test.append(df_val)
    test_csv = [[test_name, 'TEST'] for test_name in test_images]
    df_test = pd.DataFrame(test_csv, columns=['name', 'split'])
    df_val_test = df_val_test.append(df_test)
    path_save_csv_val_test = os.path.join(target_dir, val_test_csv_name)
    df_val_test.to_csv(path_save_csv_val_test)




def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    parser.add_argument('image_dir', metavar= 'image_dir', type=str,
                        help="""Directory containing 3d images""")

    parser.add_argument('csv_dir', metavar='csv_dir', type=str,
                        help="""Directory containing csv ground truth """)

    parser.add_argument('target_dir', metavar='target_dir', type=str,
                        help="""Directory to put the patches.csv file in """)

    parser.add_argument('train_csv_name', metavar='train_csv_name', type=str,
                        help="""name of the csv file with train patches """)

    parser.add_argument('val_test_csv_name', metavar='val_test_csv_name', type=str,
                        help="""Name of the csv file for validation and test """)

    parser.add_argument('perc_test', metavar='perc_test', type=float,
                        help="""Test percentage in the test-train split """)

    parser.add_argument('perc_val', metavar='perc_val', type=float,
                        help="""Validation percentage in the train validation split """)

    parser.add_argument('threshold',  metavar='threshold', type = int,
                        default=8,
                        help=""" Threshold to dark zones selection """)

    parser.add_argument('patch_size', metavar='patch_size', type=int,
                        help="""Size of the cubic patches""")


    return parser



if __name__ =="__main__":

    parser = get_parser()
    args = parser.parse_args()
    print args.perc_test
    print args.perc_val
    generate_csvs(args.image_dir, args.csv_dir, args.target_dir, args.train_csv_name, args. val_test_csv_name, args.perc_test, args.perc_val, args.patch_size,
                  args.threshold)