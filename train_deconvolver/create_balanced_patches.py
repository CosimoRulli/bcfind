import os
import pandas as pd
import numpy as np
import torch
import tifffile


def patch_generator(img, patch_size, perc=0.5):
    '''extract patches from an entire substack
    without an overlap of at most 50%'''
    overlap_percentage = (patch_size**3) * 0.5
    mask = np.zeros(img.shape)
    xs, ys, zs = np.where(mask == 0)
    points = sorted(zip(xs, ys, zs))
    np.random.shuffle(points)

    for x, y, z in points:
        patch_mask = mask[x:x+patch_size,
                          y:y+patch_size,
                          z:z+patch_size]
        overlap_area = np.sum(patch_mask)

        if overlap_area < overlap_percentage:
            mask[x:x+patch_size, y:y+patch_size, z:z+patch_size] = 1

            yield img[x:x+patch_size,
                      y:y+patch_size,
                      z:z+patch_size]


def patch_balancer(gt_dir, target_dir):
    '''Starting from the whole set of substacks it creates a dataset of
    patch in a balanced fashion'''
    pth_imgs = [os.path.join(gt_dir, pth_img) for pth_img in os.listdir(gt_dir)
                if pth_img.split(".")[-1] == "pth"]
    for img_path in pth_imgs:
        print img_path.split('/')[-1]
        img = torch.load(img_path)
        


if __name__ == "__main__":
    gt_dir = '/home/leonardo/workspaces/bcfind/dataset/GT/3d_gt'
    targer_dir = ''
    patch_balancer(gt_dir, targer_dir)
