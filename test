import argparse
import scipy.io as scio
import warnings

from eval_UDA import eval
from model.deeplabv2 import get_deeplab_v2
from config import cfg, cfg_from_file
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")
import numpy as np


def get_arguments():
    """
    Parse input arguments
    """

    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--pretrained_model_pth', type=str,
                        default='',
                        help='optional config file', )
    parser.add_argument('--target_modality', type=str, default='CT',
                        help='optional config file', )
    return parser.parse_args()


def main():
    #LOAD ARGS
    args = get_arguments()

    test_list_pth = None
    target_modality = args.target_modality
    print('target_modality is {}'.format(target_modality))

    if target_modality == 'CT':
        test_list_pth = '/root/autodl-tmp/data_np/datalist/test_ct.txt'
    if target_modality == 'MR':
        test_list_pth = '/root/autodl-tmp/data_np/datalist/test_mr.txt'
    with open(test_list_pth) as fp:
        rows = fp.readlines()
    testfile_list = [row[:-1] for row in rows]

    model = None
    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model = get_deeplab_v2(num_classes = cfg.NUM_CLASSES,multi_level=cfg.TRAIN.MULTI_LEVEL)

    pretrained_model_pth  = args.pretrained_model_pth
    dice_mean,dice_std,assd_mean,assd_std = eval(model,testfile_list,target_modality,pretrained_model_pth)

if __name__ == '__main__':
    main()
