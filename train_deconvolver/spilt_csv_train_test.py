import argparse
import pandas as pd
import numpy as np
import os


def divide_im_names(im_names, test_perc):
    n_images = im_names.shape[0]  # 66
    n_test_im = int(n_images * test_perc)
    n_train_im = n_images - n_test_im

    print("n_images : " + str(n_images))
    print("n_test_im : " + str(n_test_im))
    print("n_train_im : " + str(n_train_im))

    train_indexes = [i for i in range(n_images)]

    np.random.seed(41)
    test_indexes = np.random.randint(0, n_images, size=n_test_im)
    test_indexes.sort()
    train_indexes = np.array(list(set(train_indexes) - set(test_indexes)))
    train_images = im_names[train_indexes]
    test_images = im_names[test_indexes]
    return train_images, test_images


def split_train_test(csv_path, test_perc, dest_dir):
    df = pd.read_csv(csv_path, comment="#",
                index_col=0, dtype={"img_name": str})
    patch_size = int(str(pd.read_csv(csv_path, nrows=1, header=None).
                                  take([0])).split("=")[-1])  # workaround
    im_names= np.unique(np.array(df.loc[:]['img_name']))
    train_images, test_images = divide_im_names(im_names, test_perc)
    train_list = []
    test_list = []
    for _, row in df.iterrows():
        im_name = row['img_name']
        if im_name in train_images:
            train_list.append(row)
        else:
            test_list.append(row)

    train_csv_path = os.path.join(dest_dir, 'train_patches.csv')
    test_csv_path = os.path.join(dest_dir, 'test_patches.csv')

    train_df = pd.DataFrame(train_list, columns=['x', 'y', 'z', 'img_name'])
    test_df = pd.DataFrame(test_list, columns=['x', 'y', 'z', 'img_name'])

    train_df.to_csv(train_csv_path)
    test_df.to_csv(test_csv_path)





def get_parser():
    parser = argparse.ArgumentParser(
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv_path', dest = 'csv_path', type=str,
                        help="""Path to the patches.csv """)

    parser.add_argument('--test_perc', dest='test_perc', type=float,
                        help="""Test percentage""")

    parser.add_argument('--dest_dir', dest='dest_dir', type=str,
                        help="""Directory where to save the new csvs""")
    return parser

if __name__ == "__main__":
    parser = get_parser()

    args = parser.parse_args()
    split_train_test(args.csv_path, args.test_perc, args.dest_dir)