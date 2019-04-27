# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import torch
from torch.utils.data import DataLoader
import argparse
import utils
from models import FC_teacher_max_p

#from models.FC_teacher_max_p import FC_teacher_max_p
from data_reader import DataReader, DataReaderWeight, DataReader_2map, DataReaderSubstack, DataReaderValidation_2map
from models.FC_teacher import FC_teacher

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
import warnings


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    parser.add_argument('train_csv_path', metavar='train_csv_path', type=str,
                        help="""path to csv patches """)

    parser.add_argument('val_test_csv_path', metavar='val_test_csv_path', type=str,
                        help="""path to val/train csv""")

    parser.add_argument('img_dir', metavar='img_dir', type=str,
                        help="""Directory contaning the collection
                        of pth images""")

    parser.add_argument('gt_dir', metavar='gt_dir', type=str,
                        help="""Directory contaning the collection
                        of pth gt images""")

    parser.add_argument('weight_dir', metavar='weight_dir', type=str,
                        help="""Directory contaning the collection
                        of weighted_map""")

    parser.add_argument('centers_dir', metavar='centers_dir', type=str,
                        help="""Directory contaning the collection
                            of gt markers""")

    parser.add_argument('model_save_path', metavar='model_save_path', type=str,
                        help="""directory where the models will be saved""")

    parser.add_argument('device', metavar='device', type=str,
                        help="""device to use during training
                        and validation phase, e.g. cuda:0""")

    parser.add_argument('-w', '--weight', dest='soma_weight', type=float,
                        default=1.0,
                        help=""" wheight for class soma cell""")

    parser.add_argument('--special_loss', dest='special_loss',
                        action='store_true',
                        help="""whether to use the training with
                        a special pixel-wise loss set to true""")

    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int,
                        default=8,
                        help="""batch size, integer number""")

    parser.add_argument('-t', '--n_workers', dest='n_workers', type=int,
                        default=4,
                        help="""number of workers for data loader""")

    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=100,
                        help="""number of training epochs""")

    parser.add_argument('-k', '--kernel_size', dest='kernel_size', type=int,
                        default=3,
                        help="""size of the cubic kernel, provide only an integer e.
                        g. kernel_size 3""")

    parser.add_argument('-f', '--initial_filters', dest='initial_filters',
                        type=int,
                        default=4,
                        help="""Number of filters in the initial conv layer""")

    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.001,
                        help="""learning rate""")

    parser.add_argument('-l', '--root_log_dir', dest='root_log_dir', type=str,
                        default=None,
                        help="""log directory for tensorbard""")

    parser.add_argument('-n', '--name_dir', dest='name_dir', type=str,
                        default=None,
                        help="""name of the directory where  model will
                        be stored""")

    parser.set_defaults(special_loss=False)
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


def criterium_to_save():
    return True


def print_and_save_losses(losses, metrics, writer, epoch):
    for phase in ['train', 'validation']:
        losses[phase] = torch.tensor(losses[phase]).mean().item()
        print '{} loss: {:.3f} '.format(phase, losses[phase])
        writer.add_scalars(
            'losses',
            {phase: losses[phase]},
            global_step=epoch
        )

    for key in metrics.keys():
        metrics[key] = torch.tensor(metrics[key]).float().mean().item()
        print 'Validation {} : {:.3f} '.format(key, metrics[key])
        writer.add_scalars(
            'metrics',
            {key: metrics[key]},
            global_step = epoch
        )


if __name__ == "__main__":
    parser = get_parser()
    additional_namespace_arguments(parser)
    args = parser.parse_args()

    torch.manual_seed(9999)
    sigmoid = nn.Sigmoid()
    # tensorboard configuration
    default_log_dir = "/home/rulli_scommegna/tensorboard_log"
    datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    log_base = args.root_log_dir if args.root_log_dir else default_log_dir
    name_dir = args.name_dir if args.name_dir else datestring
    log_dir = os.path.join(log_base, name_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # create a directory for models
    # datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    name_dir = args.name_dir if args.name_dir else datestring
    model_save_path = os.path.join(args.model_save_path, name_dir)
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    train_df = pd.read_csv(args.train_csv_path, comment="#",
                                  index_col=0, dtype={"img_name": str})
    patch_size = int(str(pd.read_csv(args.train_csv_path, nrows=1, header=None).
                         take([0])).split("=")[-1])  # workaround
    print 'patch_size' + str(patch_size)
    val_test_dataframe = pd.read_csv(args.val_test_csv_path, comment="#",
                                  index_col=0, dtype={"name": str})
    val_df = val_test_dataframe[(val_test_dataframe['split'] == 'VAL')]

    if args.special_loss:
        train_dataset = DataReader_2map(args.img_dir, args.gt_dir,
                                        args.weight_dir,
                                        args.weight_dir,
                                        train_df, patch_size)

        validation_dataset = DataReaderValidation_2map(args.img_dir,
                                                       args.gt_dir,
                                                       args.centers_dir,
                                                       args.weight_dir,
                                                       args.weight_dir,
                                                       val_df)
    else:
        train_dataset = DataReaderWeight(args.img_dir, args.gt_dir,
                                         args.weight_dir,
                                         train_df, patch_size)

        validation_dataset = DataReaderSubstack(args.img_dir,
                                                args.gt_dir,
                                                args.centers_dir,
                                                args.weight_dir,
                                                val_df)

    train_loader = DataLoader(train_dataset, args.batch_size,
                              shuffle=True, num_workers=args.n_workers)

    validation_loader = DataLoader(validation_dataset, 1,
                                   shuffle=False, num_workers=args.n_workers)

    # input_size = (args.patch_size, args.patch_size, args.patch_size)

    model = FC_teacher_max_p.FC_teacher_max_p(n_filters=args.initial_filters,
                             k_conv=args.kernel_size).to(args.device)
    model.apply(utils.weights_init)
    # loss = nn.CrossEntropyLoss()
    # loss = nn.BCELoss()
    # pos_w = torch.Tensor([1, args.soma_weight])
    #loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ############
    # Main loop
    ############
    print("Starting main loop")

    losses = {}
    losses['train'] = []
    losses['validation'] = []

    metrics = {}

    best_f1 = -1
    best_epoch = 0

    for epoch in range(args.epochs):
        print("epoch [{}/{}]".format(epoch, args.epochs))
####################
        print 'Train Phase'

        model.train()
        losses['train'] = []

        dataset  = train_dataset
        total_iterations = len(dataset) // args.batch_size + 1

        progress_bar = tqdm(enumerate(train_loader),
                            total=total_iterations)
        for idx, patches in progress_bar:

            if args.special_loss:
                img_patches, gt_patches, wmap, no_wmap = patches

                weighted_map = img_patches.clone()

                weighted_map[wmap.byte()] = args.soma_weight
                weighted_map[no_wmap.byte()] = 0

                weighted_map_temp = weighted_map.clone()
                weighted_map += 1

            else:
                img_patches, gt_patches, mask = patches
                weighted_map = (mask * (args.soma_weight - 1)) + 1

            img_patches = img_patches.to(args.device)
            gt_patches = gt_patches.to(args.device)
            weighted_map = weighted_map.to(args.device)

            with torch.set_grad_enabled(True):
                model.zero_grad()

                model_output = model(img_patches)
                calc_loss = torch.mean( weighted_map.view(-1) *
                                        F.binary_cross_entropy_with_logits(
                                            model_output.view(-1),
                                            gt_patches.view(-1), reduce='none'))

                # calc_loss = F.binary_cross_entropy_with_logits(model_output.view(-1),
                #                                                gt_patches.view(-1),
                #                                                pos_weight=weighted_map.view(-1))

                losses['train'].append(calc_loss)

                calc_loss.backward()
                optimizer.step()

        print '\nValidation Phase'
        model.eval()
        losses['validation'] = []
        dataset = validation_dataset

        total_iterations = len(dataset) // args.batch_size + 1

        # progress_bar = enumerate(validation_loader)

        for idx, batch in enumerate(validation_loader):
            # img, gt, mask, centers_df = patches

            if args.special_loss:
                img, gt, mask, no_weight_mask, centers_df = batch

                weighted_map = img_patches.clone()
                
                weighted_map[wmap.byte()] = args.soma_weight
                weighted_map[no_wmap.byte()] = 0

                weighted_map_temp = weighted_map.clone()
                weighted_map += 1
            else:
                img, gt, mask, centers_df = batch
                weighted_map = (mask * (args.soma_weight - 1)) + 1

            img = img.to(args.device)
            gt = gt.to(args.device)
            weighted_map = weighted_map.to(args.device)

            '''Validation Loss'''
            with torch.set_grad_enabled(False):
                #model.zero_grad()
                model_output = model(adapt_dimension(img))

                gt_ad = adapt_dimension(gt)
                flat_gt = gt_ad.contiguous().view(-1)
                calc_loss = torch.mean(weighted_map.view(-1) *
                                        F.binary_cross_entropy_with_logits(
                                            model_output.view(-1),
                                            flat_gt, reduce='none'))

                # calc_loss = F.binary_cross_entropy_with_logits(model_output.view(-1),
                #                                                flat_gt,
                #                                                pos_weight=adapt_dimension(weighted_map).contiguous().view(-1))

                losses['validation'].append(calc_loss)

                #print 'Computing accuracy'
                blockPrint()
                #ricalcolo senza adattare le dimensioni
                model_output = model(img)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    print torch.max(sigmoid(model_output))
                    precision, recall, F1, TP_inside, FP_inside, FN_inside = evaluate_metrics(sigmoid(model_output).squeeze(), centers_df.squeeze(), args)
                enablePrint()

                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['F1'] = F1

                if F1 > best_f1:
                    best_f1  = F1
                    best_epoch = epoch
                    file_name = os.path.join(args.model_save_path, args.name_dir,  "best.txt")
                    with open(file_name, "w") as f:
                        f.write("best epoch: "+str(epoch)+"\n")
                        f.write("F1: "+str(F1))

        # after each epoch we print and save in tensorboard the losses
        print_and_save_losses(losses, metrics, writer, epoch)

        # sometimes we save the model
        if criterium_to_save():
            torch.save(model.state_dict(),
                       os.path.join(model_save_path,
                                    '{}_teacher'.format(epoch)))
