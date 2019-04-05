import os
import tifffile
import torch
import numpy as np

if __name__=="__main__":
    path_dir = "/home/cosimo/from_server_2"
    files =os.listdir(path_dir)
    for file_name in files:

        im = torch.load(os.path.join(path_dir, file_name))
        #im = im.float()/255
        im = im.numpy().astype(np.uint8)
        file_path = os.path.join(path_dir,file_name.split(".")[0] +".tif" )
        tifffile.imwrite(file_path, im, photometric='minisblack')