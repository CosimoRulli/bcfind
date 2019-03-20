import numpy as np
import pandas as pd
from skimage.filters import gaussian

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dim = (281, 245, 280)


def place_centers(mask, center_df):
    for idx, center in center_df.iterrows():
        print center.x, center.y, center.z
        mask[center.x, center.y, center.z] = 1
    return mask


def gt_builder_from_gaussian_filter(csv_path, sigma, truncate):

    df = pd.read_csv(csv_path, skipinitialspace=True, na_filter=False)
    if '#x' in df.keys():  # fix some Vaa3d garbage
        df.rename(columns={'#x': 'x'}, inplace=True)
    if '##x' in df.keys():  # fix some Vaa3d garbage
        df.rename(columns={'##x': 'x'}, inplace=True)

    centers_df = df.iloc[:, :3]
    gt_clear = np.zeros(dim)

    gt_seed = place_centers(np.copy(gt_clear), centers_df)

    gt_img = gaussian(gt_seed, sigma=sigma, truncate=truncate)
    return gt_img


if __name__ == "__main__":
    "for testing purpose"
    path = '/home/leonardo/workspaces/bcfind/dataset/GT/TomoDec13/011312-GT.marker'
    sigma = 3.5
    trunc = 1.5*sigma
    res = gt_builder_from_gaussian_filter(path, sigma, trunc)

    print res.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(res[0], res[1], res[2], color='b')
    plt.show()
