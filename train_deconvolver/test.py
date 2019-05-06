# -*- coding: utf-8 -*-


from __future__ import absolute_import

import torch
import torch.nn as nn
import os
import argparse
from bcfind.log import tee
#import utils
import numpy as np


#from data_reader import DataReader

# from models.FC_teacher import FC_teacher
from models import  FC_teacher_max_p
from models import FC_student


import tifffile
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_path  = "/home/cosimo/machine_learning_dataset/3d_images_flip_false/052204.pth"
    #img_path = "/home/leonardo/workspaces/bcfind/dataset/3d_img/073608.pth"
    img = torch.load(img_path)
    print(torch.max(img))
    img = img.float()/255
    print(torch.max(img))

    sigmoid = nn.Sigmoid()

    # model_path = "/home/cosimo/0_teacher"
    model_path = "/home/cosimo/only_quantization4bit_10"
    model = FC_teacher_max_p.FC_teacher_max_p(8, k_conv=7).to('cuda:0')
    #model = FC_student.FC_student(4, k_conv = 7).to('cuda:0')
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model = model.to('cuda:0')

    img = torch.unsqueeze(img, dim=0).float()
    img = img.to('cuda:0')
    print img.shape
    with torch.set_grad_enabled(False):
        output = sigmoid(model(img))

    #print(output)
    torch.save(output, "/home/cosimo/Desktop/output.pth")
    print(torch.max(output))
    #torch.save(output, "/home/leonardo/Desktop/output.pth")
    output*=255
    output_num = output.cpu().numpy()
    output_num = output_num.astype(np.uint8)

    hist = np.histogram(output_num, bins=range(256))
    print hist
    plt.hist(output_num.flatten())
    plt.show()

    print(np.max(output_num))

    tifffile.imwrite("/home/cosimo/Desktop/output.tif", output_num, photometric='minisblack')
    #for t in timers:
    #    tee.log(t)
