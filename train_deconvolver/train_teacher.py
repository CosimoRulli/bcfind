# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import torch
from torch.utils.data import DataLoader
import argparse
import utils
from data_reader import DataReader, DataReaderWeight, DataReader_2map
from models.FC_teacher import FC_teacher
from models.FC_teacher_max_p import FC_teacher_max_p
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from datetime import datetime
import torch.nn.functional as F

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

    parser.add_argument('weight_dir', metavar='weight_dir', type=str,
                        help="""Directory contaning the collection
                        of weighted_map""")

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

    parser.add_argument('--max_pool_version', dest='max_pool_version',
                        action='store_true',
                        help="""whether to use the teacher with
                        maxpooling layers set to true""")

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

    parser.set_defaults(max_pool_version=False)
    parser.set_defaults(special_loss=False)
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

    torch.manual_seed(9999)

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

    complete_dataframe = pd.read_csv(args.csv_path, comment="#",
                                     index_col=0, dtype={"img_name": str})
    patch_size = int(str(pd.read_csv(args.csv_path, nrows=1, header=None).
                         take([0])).split("=")[-1])  # workaround
    # 80/20 splitting rule
    train_df, val_df = train_test_split(complete_dataframe, test_size=0.2)
    if args.special_loss:
        train_dataset = DataReader_2map(args.img_dir, args.gt_dir,
                                        args.weight_dir,
                                        args.weight_dir,
                                        train_df, patch_size)

        validation_dataset = DataReader_2map(args.img_dir, args.gt_dir,
                                             args.weight_dir,
                                             args.weight_dir,
                                             val_df, patch_size)
    else:
        train_dataset = DataReaderWeight(args.img_dir, args.gt_dir,
                                         args.weight_dir,
                                         train_df, patch_size)

        validation_dataset = DataReaderWeight(args.img_dir, args.gt_dir,
                                              args.weight_dir,
                                              val_df, patch_size)

    train_loader = DataLoader(train_dataset, args.batch_size,
                              shuffle=True, num_workers=args.n_workers)

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

                if args.special_loss:
                    img_patches, gt_patches, wmap, no_wmap = patches

                    weighted_map = img_patches.clone()
                    print args.soma_weight - 1
                    weighted_map[wmap.byte()] = args.soma_weight
                    weighted_map[no_wmap.byte()] = 0

                    weighted_map_temp = weighted_map.clone()
                    weighted_map += 1
                    # #####debug#######
                    # import tifffile
                    # import numpy as np
                    # import sys
                    # for i in range(weighted_map_temp.shape[0]):
                    #     map_save = (weighted_map_temp[i] * 255).numpy().astype(np.uint8)
                    #     img_save = (img_patches[i].clone() * 255).numpy().astype(np.uint8)
                    #     gt_save = (gt_patches[i].clone() * 255).numpy().astype(np.uint8)
                    #     name = np.random.randint(1000)
                    #     tifffile.imwrite(os.path.join(
                    #         "/home/leonardo/Desktop/map_test_2/"+ str(name) +"_map.tiff"),
                    #                      map_save, photometric = 'minisblack')

                    #     tifffile.imwrite(os.path.join(
                    #         "/home/leonardo/Desktop/map_test_2/"+ str(name) +"_gt.tiff"),
                    #                      gt_save, photometric = 'minisblack')

                    #     tifffile.imwrite(os.path.join(
                    #         "/home/leonardo/Desktop/map_test_2/"+ str(name) +"_img.tiff"),
                    #                      img_save, photometric = 'minisblack')
                    #     print '####################'
                    #     # print np.sum(no_wmap.numpy())
                    #     print '####################'
                    #sys.exit()

                    # #################
                else:
                    img_patches, gt_patches, mask = patches
                    weighted_map = (mask * (args.soma_weight - 1)) + 1

                # continue

                img_patches = img_patches.to(args.device)
                gt_patches = gt_patches.to(args.device)
                # mask = mask.to(args.device)
                weighted_map = weighted_map.to(args.device)
                # FIXME mettere weighted_map al posto di mask

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

                    #loss.pos_weight = weighted_map.view(-1)
                    #calc_loss = loss(model_output.view(-1),
                    #                 gt_patches.view(-1))
                    # if args.special_loss:
                    #     calc_loss = torch.mean( weighted_map.view(-1) *
                    #                             F.binary_cross_entropy_with_logits(
                    #                                 model_output.view(-1),
                    #                                 gt_patches.view(-1), reduce='none'))
                    # else:
                    #     calc_loss  = F.binary_cross_entropy_with_logits(model_output.view(-1),
                    #                                                     gt_patches.view(-1), pos_weight=weighted_map.view(-1))

                    calc_loss  = F.binary_cross_entropy_with_logits(model_output.view(-1),
                                                                    gt_patches.view(-1), pos_weight=weighted_map.view(-1))

                    # TODO: aggiungere i pesi
                    # calc_loss = F.binary_cross_entropy_with_logits(model_output,
                    #                                               gt_patches)

                    # print calc_loss
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
