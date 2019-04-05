import torch
from bcfind import volume
from bcfind import mscd
from bcfind.scripts.eval_perf import *
import pandas as pd
from bcfind.volume import *
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.01)
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def convert_df_to_centers(gt_dataframe):
    centers_list = []
    for i in range(gt_dataframe.shape[0]):
        center = Center(gt_dataframe[i,0], gt_dataframe[i,1], gt_dataframe[i,2])
        centers_list.append(center)

    return centers_list

def evaluate_metrics(pred_tensor, gt_dataframe, args):
    #todo convertire gt_dataframe in Centers
    '''


    :param pred_tensor: Tensor of shape D, H, W
    :param gt_dataframe: Tensor of shape [n_centers, 3]
    :param args:
    :return:
    '''
    print gt_dataframe.shape

    substack = volume.SubStack('', '0')
    pred_tensor_np= pred_tensor.numpy()
    #print type(pred_tensor_np)
    #todo il file plist va creato e passato -> Aggiunto Margin:40 dentro il costruttore di Volume
    substack.load_volume_from_3D(pred_tensor_np)

    C_pred, pred_seed = mscd.ms(substack, args)
    C_true = convert_df_to_centers(gt_dataframe)
    if args.manifold_distance:
        try:
            for c in C_pred:
                c.rejected = c.distance >= args.manifold_distance
        except AttributeError:
            print('You specified a manifold distance',args.manifold_distance,'however markers file' 'is not annotated with a distance column')
    else:
        for c in C_pred:
            c.rejected = False
    if args.do_icp:
        precision,recall,F1,TP_inside,FP_inside,FN_inside = eval_perf_icp(substack,C_true,C_pred,verbose=args.verbose,errors_marker_file=None, max_cell_diameter=args.max_cell_diameter)
    else:
        precision,recall,F1,TP_inside,FP_inside,FN_inside = eval_perf(substack,C_true,C_pred,
                                                                      errors_marker_file=None,
                                                                      rp_file=None,
                                                                      verbose=args.verbose,
                                                                      max_cell_diameter=args.max_cell_diameter)


    return precision,recall,F1,TP_inside,FP_inside,FN_inside


