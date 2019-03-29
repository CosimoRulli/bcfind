# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import torch
from torch.utils.data import DataLoader
import argparse
import utils
from data_reader import DataReader
from models.FC_teacher import FC_teacher
from models.FC_teacher_max_p import FC_teacher_max_p
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from datetime import datetime


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    parser.add_argument('csv_path', metavar='csv_path', type=str,
                        help="""path for csv patches file""")

    parser.add_argument('img_dir', metavar='img_dir', type=str,
                        help="""Directory contaning the collection
                        of pth images""")

    parser.add_argument('gt_dir', metavar='gt_dir', type=str,
                        help="""Directory contaning the collection
                        of pth gt images""")

    parser.add_argument('model_save_path', metavar='model_save_path', type=str,
                        help="""directory where the models will be saved""")

    parser.add_argument('batch_size', metavar='batch_size', type=int,
                        default=8,
                        help="""batch size, integer number""")

    parser.add_argument('n_workers', metavar='n_workers', type=int,
                        default=4,
                        help="""number of workers for data loader""")

    parser.add_argument('epochs', metavar='epochs', type=int,
                        default=100,
                        help="""number of training epochs""")

    parser.add_argument('kernel_size', metavar='kernel_size', type=int,
                        default=3,
                        help="""size of the cubic kernel, provide only an integer e.
                        g. kernel_size 3""")

    parser.add_argument('max_pool_version', metavar='max_pool_version',
                        type=bool,
                        default=False,
                        help="""whether to use the teacher with
                        maxpooling layers set to true""")

    parser.add_argument('initial_filters', metavar='initial_filters', type=int,
                        default=4,
                        help="""Number of filters in the initial conv layer""")

    parser.add_argument('patch_size', metavar='patch_size', type=int,
                        default=13,
                        help="""size of the cubic patch, provide only an integer
                        e.g. patch_zie 13""")

    parser.add_argument('lr', metavar='lr', type=float,
                        default=0.001,
                        help="""learning rate""")

    parser.add_argument('device', metavar='device', type=str,
                        help="""device to use during training
                        and validation phase, e.g. cuda:0""")

    parser.add_argument('root_log_dir', metavar='root_log_dir', type=str,
                        default=None,
                        help="""log directory for tensorbard""")

    return parser


def criterium_to_save():
    return True


def print_and_save_losses(losses, writer, epoch):
    for phase in ['train', 'validation']:
        losses[phase] = torch.tensor(losses[phase]).mean().item()
        print '{} loss: {:.3f} '.format(phase, losses[phase])
        writer.add_scalars(
            'losses',
            {phase: losses[phase]},
            global_step=epoch
        )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # create a directory for models
    datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    model_save_path = os.path.join(args.model_save_path, datestring)
    os.makedirs(model_save_path)

    complete_dataframe = pd.read_csv(args.csv_path, comment="#",
                                     index_col=0, dtype={"img_name": str})
    patch_size = int(str(pd.read_csv(args.csv_path, nrows=1, header=None).
                         take([0])).split("=")[-1])  # workaround
    # 80/20 splitting rule
    train_df, val_df = train_test_split(complete_dataframe, test_size=0.2)

    train_dataset = DataReader(args.img_dir, args.gt_dir, train_df, patch_size)
    train_loader = DataLoader(train_dataset, args.batch_size,
                              shuffle=True, num_workers=args.n_workers)

    validation_dataset = DataReader(args.img_dir, args.gt_dir,
                                    val_df, patch_size)
    validation_loader = DataLoader(validation_dataset, args.batch_size,
                                   shuffle=False, num_workers=args.n_workers)

    # input_size = (args.patch_size, args.patch_size, args.patch_size)

    if args.max_pool_version:
        print 'sono max'
        print args.max_pool_version
        model = FC_teacher_max_p(n_filters=args.initial_filters,
                                 k_conv=args.kernel_size).to(args.device)
    else:
        print 'sono normale'
        print args.max_pool_version

        model = FC_teacher(k=args.kernel_size,
                           n_filters=args.initial_filters).to(args.device)

    model.apply(utils.weights_init)
    # loss = nn.CrossEntropyLoss()
    loss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # tensorboard configuration
    default_log_dir = "/home/rulli_scommegna/tensorboard_log"
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_base = args.root_log_dir if args.root_log_dir else default_log_dir
    log_dir = os.path.join(log_base, current_time)

    writer = SummaryWriter(log_dir=log_dir)

    ############
    # Main loop
    ############
    print("Starting main loop")

    losses = {}
    losses['train'] = []
    losses['validation'] = []

    for epoch in range(args.epochs):
        print("epoch [{}/{}]".format(epoch, args.epochs))

        for phase in ['train', 'validation']:

            print("Phase: {}".format(phase))

            if phase == 'train':
                model.train()
            else:
                model.eval()

            losses[phase] = []

            dataset = train_dataset if phase == 'train' else validation_dataset

            total_iterations = len(dataset) // args.batch_size + 1

            # progress_bar = enumerate(total=total_iterations)

            if phase == 'train':
                progress_bar = tqdm(enumerate(train_loader),
                                    total=total_iterations)
            else:
                progress_bar = enumerate(validation_loader)

            for idx, patches in progress_bar:
                img_patches, gt_patches = patches

                img_patches = img_patches.to(args.device)
                gt_patches = gt_patches.to(args.device)

                with torch.set_grad_enabled(phase == 'train'):
                    model.zero_grad()
                    model_output = model(img_patches)

                    # print '****************************'
                    # print model_output.shape
                    # print gt_patches.shape
                    # print '****************************'

                    # model_output_flat = model_output.view(-1)
                    # gt_patches_flat = gt_patches.view(-1)

                    # print '****************************'
                    # print model_output_flat.shape
                    # print gt_patches_flat.shape
                    # print '****************************'

                    calc_loss = loss(model_output, gt_patches)
                    print calc_loss
                    losses[phase].append(calc_loss)
                    if phase == 'train':
                        calc_loss.backward()
                        optimizer.step()

        # after each epoch we print and save in tensorboard the losses
        print_and_save_losses(losses, writer, epoch)

        # sometimes we save the model
        if criterium_to_save():
            torch.save(model.state_dict(),
                       os.path.join(model_save_path,
                                    '{}_teacher'.format(epoch)))
