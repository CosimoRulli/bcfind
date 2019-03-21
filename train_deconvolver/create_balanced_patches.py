import os
import pandas as pd
import numpy as np
import torch
import tifffile


def patch_generator_old(img, patch_size, perc=0.5):
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

        if overlap_area < overlap_percentage and (condition1 or condition2):
            mask[x:x+patch_size, y:y+patch_size, z:z+patch_size] = 1

            yield img[x:x+patch_size,
                      y:y+patch_size,
                      z:z+patch_size]


def patch_generator(img_name, patch_size, perc=0.5):
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

        if overlap_area < overlap_percentage and (condition1 or condition2):
            mask[x:x+patch_size, y:y+patch_size, z:z+patch_size] = 1

            yield img[x:x+patch_size,
                      y:y+patch_size,
                      z:z+patch_size]


def patch_balancer(gt_dir, image_dir, csv_dir, target_root_dir, patch_size):
    '''Starting from the whole set of substacks it creates a dataset of
    patch in a balanced fashion'''

    img_names = [name.split(".")[0] for name in os.listdir(image_dir)]

    for img_name in img_names:
        print 'magi'

    ########## old
    pth_imgs = [os.path.join(gt_dir, pth_img) for pth_img in os.listdir(gt_dir)
                if pth_img.split(".")[-1] == "pth"]
    for img_path in pth_imgs:
        print img_path.split('/')[-1]
        img = torch.load(img_path)
        for patch in patch_generator(img, patch_size):
            print 'magi'


if __name__ == "__main__":
    gt_dir = '/home/leonardo/workspaces/bcfind/dataset/GT/3d_gt'
    targer_dir = ''
    patch_balancer(gt_dir, targer_dir)
