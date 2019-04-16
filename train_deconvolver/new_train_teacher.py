# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import torch
from torch.utils.data import DataLoader
import argparse
import utils
from data_reader import DataReader, DataReaderWeight, DataReaderSubstack
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

    parser.add_argument('marker_dir', metavar='marker_dir', type=str,
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
    return parser


def additional_namespace_arguments(parser):
    '''
    parser.add_argument('-r', '--hi_local_max_radius', metavar='r', dest='hi_local_max_radius',
                        action='store', type=float, default=6,
                        help='Radius of the seed selection ball (r)')

    parser.add_argument('-fl', '--floating_point', dest='floating_point', action='store_true',
                        help='If true, cell centers are saved in floating point.')

    parser.add_argument('-m', '--min_second_threshold', metavar='min_second_threshold', dest='min_second_threshold',
                        action='store', type=int, default=15,
                        help="""If the foreground (second threshold in multi-Kapur) is below this value
                                   then the substack is too dark and assumed to contain no soma""")
    parser.add_argument('-loc', '--local', dest='local', action='store_true',
                        help='Perform local processing by dividing the volume in 8 parts.')
    # parser.set_defaults(local=True)
    parser.add_argument('-ts', '--seeds_filtering_mode', dest='seeds_filtering_mode',
                        action='store', type=str, default='soft',
                        help="Type of seed selection ball ('hard' or 'soft')")
    parser.add_argument('-R', '--mean_shift_bandwidth', metavar='R', dest='mean_shift_bandwidth',
                        action='store', type=float, default=5.5,
                        help='Radius of the mean shift kernel (R)')
    parser.add_argument('-s', '--save_image', dest='save_image', action='store_true',
                        help='Save debugging substack for visual inspection (voxels above threshold and colorized clusters).')
    parser.add_argument('-M', '--max_expected_cells', metavar='max_expected_cells', dest='max_expected_cells',
                        action='store', type=int, default=10000,
                        help="""Max number of cells that may appear in a substack""")
    parser.add_argument('-p', '--pair_id', dest='pair_id',
                        action='store', type=str,
                        help="id of the pair of views, e.g 000_090. A folder with this name will be created inside outdir/substack_id")
    parser.add_argument('-D', '--max_cell_diameter', dest='max_cell_diameter', type=float, default=16.0,
                        help='Maximum diameter of a cell')
    parser.add_argument('-d', '--manifold-distance', dest='manifold_distance', type=float, default=None,
                        help='Maximum distance from estimated manifold to be included as a prediction')
    parser.add_argument('-c', '--curve', dest='curve', action='store_true', help='Make a recall-precision curve.')
    parser.add_argument('-g', '--ground_truth_folder', dest='ground_truth_folder', type=str, default=None,
                        help='folder containing merged marker files (for multiview images)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Verbose output.')
    parser.add_argument('--do_icp', dest='do_icp', action='store_true',
                        help='Use the ICP matching procedure to evaluate the performance')

    parser.add_argument('-e', dest='evaluation', action='store_true')
    '''
    parser.set_defaults(hi_local_max_radius=6)
    #parser.set_defaults()
    parser.set_defaults(min_second_threshold =15)
    parser.set_defaults(mean_shift_bandwidth = 5.5)
    parser.set_defaults(seeds_filtering_mode='soft')
    parser.set_defaults(max_expected_cells = 10000)
    parser.set_defaults(max_cell_diameter = 16.0)
    parser.set_defaults(verbose = False)
    parser.set_defaults(save_image=False)
    parser.set_defaults(evaluation=True)
    parser.set_defaults(do_icp= True)
    parser.set_defaults(manifold_distance = 40)


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
    os.makedirs(model_save_path)

    ##load train dataframe
    train_dataframe = pd.read_csv(args.train_csv_path, comment="#",
                                  index_col=0, dtype={"img_name": str})
    patch_size = int(str(pd.read_csv(args.train_csv_path, nrows=1, header=None).
                         take([0])).split("=")[-1])  # workaround
    print 'patch_size' + str(patch_size)
    val_test_dataframe = pd.read_csv(args.val_test_csv_path, comment="#",
                                  index_col=0, dtype={"name": str})
    val_dataframe = val_test_dataframe[(val_test_dataframe['split']=='VAL')]

    '''
    # 80/20 splitting rule
    train_df, val_df = train_test_split(train_dataframe, test_size=0.2)
    '''
    train_dataset = DataReaderWeight(args.img_dir, args.gt_dir,
                                     args.weight_dir,
                                     train_dataframe, patch_size)
    train_loader = DataLoader(train_dataset, args.batch_size,
                              shuffle=True, num_workers=args.n_workers)

    validation_dataset = DataReaderSubstack(args.img_dir, args.gt_dir,args.marker_dir, args.weight_dir, val_dataframe )

    validation_loader = DataLoader(validation_dataset, 1,
                                   shuffle=False, num_workers=args.n_workers)

    # input_size = (args.patch_size, args.patch_size, args.patch_size)



    print 'Using max pool version'
    model = FC_teacher_max_p(n_filters=args.initial_filters,
                                 k_conv=args.kernel_size).to(args.device)


    model.apply(utils.weights_init)
    # loss = nn.CrossEntropyLoss()
    # loss = nn.BCELoss()
    # pos_w = torch.Tensor([1, args.soma_weight])
    # loss = nn.BCEWithLogitsLoss()
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
        print 'Train Phase'

        model.train()
        losses['train'] = []

        dataset  = train_dataset
        total_iterations = len(dataset) // args.batch_size + 1

        progress_bar = tqdm(enumerate(train_loader),
                            total=total_iterations)
        for idx, patches in progress_bar:

            img_patches, gt_patches, mask = patches

            img_patches = img_patches.to(args.device)
            gt_patches = gt_patches.to(args.device)
            mask = mask.to(args.device)

            with torch.set_grad_enabled(True):
                model.zero_grad()

                model_output = model(img_patches)
                weighted_map = (mask * (args.soma_weight - 1)) + 1

                calc_loss = F.binary_cross_entropy_with_logits(model_output.view(-1),
                                                               gt_patches.view(-1),
                                                               pos_weight=weighted_map.view(-1))

                losses['train'].append(calc_loss)

                calc_loss.backward()
                optimizer.step()




        print '\nValidation Phase'
        model.eval()
        losses['validation'] = []
        dataset = validation_dataset

        total_iterations = len(dataset) // args.batch_size + 1

        progress_bar = enumerate(validation_loader)

        for idx, patches in progress_bar:
            img, gt, mask, centers_df = patches

            img= img.to(args.device)
            gt = gt.to(args.device)
            mask = mask.to(args.device)

            '''Validation Loss'''
            with torch.set_grad_enabled(False):
                #model.zero_grad()
                model_output = model(adapt_dimension(img))
                weighted_map = (mask * (args.soma_weight - 1)) + 1

                #print model_output.shape
                gt_ad = adapt_dimension(gt)
                flat_gt = gt_ad.contiguous().view(-1)
                calc_loss = F.binary_cross_entropy_with_logits(model_output.view(-1),
                                                               flat_gt,
                                                               pos_weight=adapt_dimension(weighted_map).contiguous().view(-1))

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

        print_and_save_losses(losses,metrics,  writer, epoch)

        # sometimes we save the model
        if criterium_to_save():
            torch.save(model.state_dict(),
                       os.path.join(model_save_path,
                                    '{}_teacher'.format(epoch)))
