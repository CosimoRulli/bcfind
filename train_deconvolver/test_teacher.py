# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import torch
from torch.utils.data import DataLoader
import argparse
import utils
from data_reader import DataReaderSubstackTest
from models.FC_teacher import FC_teacher
from models import FC_teacher_max_p
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from datetime import datetime
import torch.nn.functional as F
from utils import *
import sys
import numpy as np
import warnings

from bcfind.log import tee









def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_path', metavar='model_path', type=str,
                        help="""path to the .pth file that represent the model to test""")

    parser.add_argument('val_test_csv_path', metavar='val_test_csv_path', type=str,
                        help="""path to val/train csv""")

    parser.add_argument('img_dir', metavar='img_dir', type=str,
                        help="""Directory contaning the collection
                        of pth images""")

    parser.add_argument('gt_dir', metavar='gt_dir', type=str,
                        help="""Directory contaning the collection
                        of pth gt images""")
    '''
    parser.add_argument('weight_dir', metavar='weight_dir', type=str,
                        help="""Directory contaning the collection
                        of weighted_map""")
    '''
    parser.add_argument('marker_dir', metavar='marker_dir', type=str,
                        help="""Directory contaning the collection
                            of gt markers""")

    parser.add_argument('save_path', metavar='save_path', type =str,
                        help ="directory where to save results")

    parser.add_argument('test_name', metavar='test_name', type = str,
                        help =" name of the test e.g. weight1.5_f8_teacher")

    parser.add_argument('device', metavar='device', type=str,
                        help="""device to use during training
                        and validation phase, e.g. cuda:0""")


    parser.add_argument('-t', '--n_workers', dest='n_workers', type=int,
                        default=4,
                        help="""number of workers for data loader""")


    parser.add_argument('-k', '--kernel_size', dest='kernel_size', type=int,
                        default=3,
                        help="""size of the cubic kernel, provide only an integer e.
                        g. kernel_size 3""")

    parser.add_argument('-f', '--initial_filters', dest='initial_filters',
                        type=int,
                        default=4,
                        help="""Number of filters in the initial conv layer""")



    return parser


def additional_namespace_arguments(parser):

    parser.set_defaults(hi_local_max_radius=6)
    parser.set_defaults(min_second_threshold=15)
    parser.set_defaults(mean_shift_bandwidth=5.5)
    parser.set_defaults(seeds_filtering_mode='soft')
    parser.set_defaults(max_expected_cells=10000)
    parser.set_defaults(max_cell_diameter=16.0)
    parser.set_defaults(verbose=False)
    parser.set_defaults(save_image=False)
    parser.set_defaults(evaluation=True)
    parser.set_defaults(do_icp=True)
    parser.set_defaults(manifold_distance=40)

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


if __name__ =="__main__":
    parser = get_parser()
    additional_namespace_arguments(parser)
    args = parser.parse_args()

    save_path = os.path.join(args.save_path, args.test_name + '.txt')
    if os.path.isfile(save_path):
        raise ValueError("The result file already exists, provide a new save_path")


    torch.manual_seed(9999)
    sigmoid = nn.Sigmoid()

    val_test_dataframe = pd.read_csv(args.val_test_csv_path, comment="#",
                                     index_col=0, dtype={"name": str})
    test_dataframe = val_test_dataframe[(val_test_dataframe['split'] == 'TEST')]

    test_dataset = DataReaderSubstackTest(args.img_dir, args.gt_dir, args.marker_dir, test_dataframe)

    test_loader = DataLoader(test_dataset, 1,shuffle=False, num_workers=args.n_workers)

    model = FC_teacher_max_p.FC_teacher_max_p(args.initial_filters, k_conv = args.kernel_size).to(args.device)

    model.load_state_dict(torch.load(args.model_path, map_location=args.device ))

    print 'Starting test'
    model.eval()

    img_names = []
    precisions = []
    recalls = []
    F1s = []


    total_iterations = len(test_dataset)

    progress_bar = enumerate(test_loader)

    timers = [FC_teacher_max_p.forward_time_teacher]

    for idx, batch  in progress_bar:
        img, gt, centers_df, img_name   = batch
        img = img.to(args.device)
        gt = gt.to(args.device)
        with torch.set_grad_enabled(False):
            print img_name
            model_output = model(img)
            blockPrint()
            with warnings.catch_warnings():

                warnings.simplefilter('ignore')

                precision, recall, F1, TP_inside, FP_inside, FN_inside = evaluate_metrics(sigmoid(model_output).squeeze(0),
                                                                                      centers_df.squeeze(0), args)
            enablePrint()
            print precision
            print recall
            print F1
            img_names.append(img_name)
            precisions.append(precision)
            recalls.append(recall)
            F1s.append(F1)

    img_names.append('mean')
    img_names.append('var')


    mean_precision = np.mean(np.array(precisions))
    precisions.append(mean_precision)
    mean_recall = np.mean(np.array(recalls))
    recalls.append(mean_recall)
    mean_F1 = np.mean(np.array(F1s))
    F1s.append(mean_F1)

    var_precision= np.var(np.array(precisions))
    precisions.append(var_precision)

    var_recall = np.var(np.array(recalls))
    recalls.append(var_recall)

    var_F1 = np.var(np.array(F1s))
    F1s.append(var_F1)

    dict  = {'img_name' : img_names, 'precision' : precisions, 'recall' : recalls, 'F1': F1s }

    df = pd.DataFrame(dict, columns=['img_name','precision', 'recall', 'F1' ])

    df.to_csv(save_path)

    for t in timers:
        tee.log(t)

