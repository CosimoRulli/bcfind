import numpy as np
from skimage.filters import gaussian


def place_centers(self, mask, center_list):
    pass


def gt_builder_from_gaussian_filter(self, json_path, sigma, truncate):
    '''costruire tramite json una zersos 3d con i centri messi a uno '''

    centers_list = [0, 0, 0]  # va calcolata dal json

    gt_clear = np.zeros(1000, 1000, 1000)

    gt_seed = place_centers(gt_clear, centers_list)

    gt_img = gaussian(gt_seed, sigma=sigma, truncate=truncate)
    return gt_img
