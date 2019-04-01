from __future__ import absolute_import

import torch
import os
import argparse

import utils
import numpy as np

from data_reader import DataReader

# from models.FC_teacher import FC_teacher
from models.FC_teacher_max_p import FC_teacher_max_p
import tifffile
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd


if __name__ == "__main__":
    # img_path  = "/home/cosimo/machine_learning_dataset/3DImages/052204.pth"
    img_path = "/home/leonardo/workspaces/bcfind/dataset/3d_img/052204.pth"
    img = torch.load(img_path)
    print(torch.max(img))
    img = img.float()/255
    print(torch.max(img))

    # model_path = "/home/cosimo/0_teacher"
    model_path = "/home/leonardo/workspaces/bcfind/dataset/models/1_teacher"
    model = FC_teacher_max_p(4).to('cuda:0')
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model = model.to('cuda:0')
    img = torch.unsqueeze(img, dim=0).float()
    img = img.to('cuda:0')
    print img.shape
    with torch.set_grad_enabled(False):
        output = model(img)

    #print(output)
    print(torch.max(output))
    output*=255
    output_num = output.cpu().numpy()
    output_num = output_num.astype(np.uint8)
    print(np.max(output_num))
    tifffile.imwrite("/home/leonardo/Desktop/output.tif", output_num, photometric='minisblack')
