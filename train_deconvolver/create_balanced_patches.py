import os
import pandas as pd
import numpy as np
import torch
import tifffile


def patch_generator_old(img, patch_size, perc=0.5):
    '''extract patches from an entire substack
    with an overlap of at most 50%'''
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

def read_single_csv(csv_img_path):
    df = pd.read_csv(csv_img_path, skipinitialspace=True, na_filter=False)
    if '#x' in df.keys():  # fix some Vaa3d garbage
        df.rename(columns={'#x': 'x'}, inplace=True)
    if '##x' in df.keys():  # fix some Vaa3d garbage
        df.rename(columns={'##x': 'x'}, inplace=True)

    df.rename(columns={'x': 'z', 'z': 'x'}, inplace=True) #todo fattelo rispiegare da leo perchè te lo sei scordato, magi-like
    centers_df = df.iloc[:, :3]
    centers = [[center.x, center.y, center.z] for _,center in centers_df.iterrows()]

    return centers


def contains_centers(coordinates, csv_img_path, patch_size, min_centers): #todo da chiarire il concetto di "contiene il centro" : almeno un pixel della palla \\
                                                                          # di raggio 5 oppure il centro del gt (per ora la seconda)
    coordinates = np.array(coordinates)
    centers = read_single_csv(csv_img_path)
    n_centers_contained =0
    for center in centers:
        contained = (coordinates<center).all() and ((coordinates + patch_size ) > center).all()
        if contained:
            n_centers_contained+=1
            if n_centers_contained >= min_centers:
                return True
    return False

def patch_generator(img_name, patch_size, gt_dir, csv_dir, perc=0.5):
    '''extract patches from an entire substack
    without an overlap of at most 50%'''
    overlap_percentage = (patch_size**3) * 0.5

    gt_img_path = os.path.join(gt_dir, img_name) +".pth"
    csv_img_path = os.path.join(csv_dir,img_name)+"-GT.marker"
    torch_gt_image = torch.load(gt_img_path)
    gt_image = torch_gt_image.numpy()

    mask = np.zeros(gt_image.shape)

    xs, ys, zs = np.where(mask == 0)
    points = sorted(zip(xs, ys, zs))
    np.random.shuffle(points)

    for x, y, z in points: #lento provare tutti i punti, possibilta' di introdurre un'altra condiione di arresto
        patch_mask = mask[x:x+patch_size,
                          y:y+patch_size,
                          z:z+patch_size]
        overlap_area = np.sum(patch_mask)
        contains_centers = contains_centers([x,y,z], csv_img_path, patch_size,1 )
        if overlap_area < overlap_percentage and (contains_centers or condition2):
            mask[x:x+patch_size, y:y+patch_size, z:z+patch_size] = 1

            yield img[x:x+patch_size,
                      y:y+patch_size,
                      z:z+patch_size]


def patch_balancer(gt_dir, image_dir, csv_dir, target_root_dir, patch_size):
    '''Starting from the whole set of substacks it creates a dataset of
    patch in a balanced fashion'''

    img_names = [name.split(".")[0] for name in os.listdir(image_dir)]

    for img_name in img_names:
        patch_generator(img_name, patch_size, gt_dir, csv_dir, 0.5 )

    #todo mancano i contatori per le condizioni
    ########## old
    '''
    pth_imgs = [os.path.join(gt_dir, pth_img) for pth_img in os.listdir(gt_dir)
                if pth_img.split(".")[-1] == "pth"]
    for img_path in pth_imgs:
        print img_path.split('/')[-1]
        img = torch.load(img_path)
        for patch in patch_generator(img, patch_size):
            print 'magi'
    '''

if __name__ == "__main__":
    gt_dir = '/home/leonardo/workspaces/bcfind/dataset/GT/3d_gt'
    targer_dir = ''
    patch_size = 64
    patch_balancer(gt_dir, targer_dir, patch_size = patch_size)
