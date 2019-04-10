from __future__ import absolute_import
import torch

from bcfind import volume
from bcfind import mscd
from bcfind.scripts.eval_perf import eval_perf_icp
import pandas as pd
from bcfind.volume import *
import numpy as np
import argparse

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

def adapt_dimension(tensor):
    #new_tensor = tensor.squeeze()
    dim = torch.tensor(tensor.shape)
    new_dim = dim // 2 *2
    new_tensor = tensor[:, :new_dim[1], :new_dim[2], :new_dim[3]]
    return new_tensor


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
    #print gt_dataframe.shape
    substack_id = '0'

    substack_dict = {substack_id: {'Files': 'dummy files', 'Height': pred_tensor.shape[1], 'Width': pred_tensor.shape[2], 'Depth': pred_tensor.shape[0]}}
    plist = {'Height': pred_tensor.shape[1], 'Width': pred_tensor.shape[2], 'Depth': pred_tensor.shape[0], 'Margin': 40, 'SubStacks': substack_dict}

    substack = volume.SubStack('', substack_id, plist)
    pred_tensor_np= pred_tensor.cpu().numpy()
    #print type(pred_tensor_np)
    #todo il file plist va creato e passato -> Aggiunto Margin:40 dentro il costruttore di Volume
    substack.load_volume_from_3D(pred_tensor_np)
    res = mscd.ms(substack, args)
    if res!= None:
        C_pred= res[0]
        pred_seed = res[1]
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
    else:
        precision= recall=F1=TP_inside= FP_inside= FN_inside=0


    return precision,recall,F1,TP_inside,FP_inside,FN_inside


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--hi_local_max_radius', metavar='r', dest='hi_local_max_radius',
                        action='store', type=float, default=6,
                        help='Radius of the seed selection ball (r)')

    parser.add_argument('-f', '--floating_point', dest='floating_point', action='store_true',
                        help='If true, cell centers are saved in floating point.')

    parser.add_argument('-m', '--min_second_threshold', metavar='min_second_threshold', dest='min_second_threshold',
                        action='store', type=int, default=15,
                        help="""If the foreground (second threshold in multi-Kapur) is below this value
                               then the substack is too dark and assumed to contain no soma""")
    parser.add_argument('-l', '--local', dest='local', action='store_true',
                        help='Perform local processing by dividing the volume in 8 parts.')
    # parser.set_defaults(local=True)
    parser.add_argument('-t', '--seeds_filtering_mode', dest='seeds_filtering_mode',
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
    parser.add_argument('-out', metavar='outdir', )
    parser.set_defaults(save_image=False)
    parser.set_defaults(evaluation=False)

    return parser


if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    im = torch.load("/home/cosimo/Desktop/output.pth").float()
    print(torch.max(im) )
    centers_df = pd.read_csv("/home/cosimo/machine_learning_dataset/GT/TomoDec13/042908-GT.marker", usecols=[0, 1, 2])
    if '#x' in centers_df.keys():  # fix some Vaa3d garbage
        centers_df.rename(columns={'#x': 'x'}, inplace=True)
        centers_df.rename(columns={'x': 'z', 'z': 'x'}, inplace=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        print(evaluate_metrics(im.squeeze(), torch.Tensor(centers_df.values).squeeze(), args))