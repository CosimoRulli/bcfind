
import torch
from torch.utils.data import DataLoader
import argparse
import utils
from data_reader import DataReader
from models.FC_teacher import FC_teacher
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)
    parser.add_argument('train_dir', metavar='train_dir', type=str,
                        help="""Directory contaning the collection
                        of pkl training file""")

    parser.add_argument('validation_dir', metavar='validation_dir', type=str,
                        help="""Directory contaning the collection
                        of pkl validation file""")

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

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    train_dataset = DataReader(args.train_dir)
    train_loader = DataLoader(train_dataset, args.batch_size,
                              shuffle=True, num_workers=args.n_workers)

    validation_dataset = DataReader(args.validation_dir)
    validation_loader = DataLoader(validation_dataset, args.batch_size,
                                   shuffle=False, num_workers=args.n_workers)

    input_size = (args.patch_size, args.patch_size, args.patch_size)
    model = FC_teacher(k=args.kernel_size,
                       n_filters=args.initial_filters,
                       input_size=input_size)

    model.apply(utils.weights_init())
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ############
    # Main loop
    ############
    print("Starting main loop")

    losses = {}
    losses['train'] = {}
    losses['validation'] = {}

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
                progress_bar = tqdm(enumerate(train_loader,
                                              total=total_iterations))
            else:
                progress_bar = enumerate(validation_loader)

            for patch in progress_bar:
                patch = patch.to(args.device)

                with torch.set_grad_enabled(phase == 'train'):
                    model.zero_grad()
                    
                    pass





