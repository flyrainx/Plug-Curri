import argparse
import os
import os.path as osp
import pprint
import random
import warnings
import numpy as np
import yaml
import torch
from torch import nn
from torch.utils import data
from data_loader import CTDataset, MRDataset
from model.deeplabv2 import get_deeplab_v2
from config import cfg, cfg_from_file
from train_UDA import train_domain_adaptation
from utils import avg_pseudo_entropy

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default=' ',
                        help='optional config file', )
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    return parser.parse_args()


def main():

    args = get_arguments()
    print('Called with args')
    print(args)

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)


    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT,cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)

    #tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR,exist_ok=True)

        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE  = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ''
    print('Using config:')
    pprint.pprint(cfg)

    # Initialization
    _init_fn = None

    torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.TRAIN.RANDOM_SEED)
    np.random.seed(cfg.TRAIN.RANDOM_SEED)
    random.seed(cfg.TRAIN.RANDOM_SEED)
    # torch.backends.cudnn.deterministic = True

    def _init_fn(worker_id):
        np.random.seed(cfg.TRAIN.RANDOM_SEED+worker_id)
    
    if os.environ.get('ADVENT_DRY_RUN','0') == '1' :
        return

    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'

    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM, map_location=torch.device('cpu'))
        if 'DeepLab_resnet_pretrained' in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)

        print('Model loaded')

    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")

    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)
        
    curriculum_stage = 0
    entropy_thresholds = [0.02, 0.012, 0.011, 0.01]

    for epoch in range(cfg.TRAIN.EPOCH):

        # DataLoaders
        train_mr_data_pth = '/root/autodl-tmp/data_np/datalist/train_mr.txt'
        train_mr_gt_pth   = '/root/autodl-tmp/data_np/datalist/gt_train_mr.txt'
        train_ct_data_pth = '/root/autodl-tmp/data_np/datalist/train_ct.txt'
        train_ct_gt_pth   = '/root/autodl-tmp/data_np/datalist/gt_train_ct.txt'
        val_mr_data_pth   = '/root/autodl-tmp/data_np/datalist/val_mr.txt'
        val_ct_data_pth   = '/root/autodl-tmp/data_np/datalist/val_ct.txt'
        val_mr_gt_pth     = '/root/autodl-tmp/data_np/datalist/gt_val_mr.txt'
        val_ct_gt_pth     = '/root/autodl-tmp/data_np/datalist/gt_val_ct.txt'

        transforms = None
        img_mean = cfg.TRAIN.IMG_MEAN

        if cfg.SOURCE == 'MR':
            strain_dataset = MRDataset(data_pth=train_mr_data_pth, gt_pth=train_mr_gt_pth,
                                       img_mean=img_mean, transform=transforms, curriculum_stage)
            strain_loader = data.DataLoader(strain_dataset,
                                            batch_size=cfg.TRAIN.BATCH_SIZE,
                                            num_workers=cfg.NUM_WORKERS,
                                            shuffle=True,
                                            pin_memory=True,
                                            worker_init_fn=_init_fn)
            trgtrain_dataset = CTDataset(data_pth=train_ct_data_pth, gt_pth=train_ct_gt_pth,
                                         img_mean=img_mean, transform=transforms)
            trgtrain_loader = data.DataLoader(trgtrain_dataset,
                                              batch_size=cfg.TRAIN.BATCH_SIZE,
                                              num_workers=cfg.NUM_WORKERS,
                                              shuffle=True,
                                              pin_memory=True,
                                              worker_init_fn=_init_fn)
            sval_dataset = MRDataset(data_pth=val_mr_data_pth, gt_pth=val_mr_gt_pth,
                                     img_mean=img_mean, transform=transforms)
            sval_loader = data.DataLoader(sval_dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE,
                                          num_workers=cfg.NUM_WORKERS,
                                          shuffle=True,
                                          pin_memory=True,
                                          worker_init_fn=_init_fn)

        elif cfg.SOURCE == 'CT':

            strain_dataset = CTDataset(data_pth=train_ct_data_pth, gt_pth=train_ct_gt_pth,
                                       img_mean=img_mean, transform=transforms, curriculum_stage)
            strain_loader = data.DataLoader(strain_dataset,
                                            batch_size=cfg.TRAIN.BATCH_SIZE,
                                            num_workers=cfg.NUM_WORKERS,
                                            shuffle=True,
                                            pin_memory=True,
                                            worker_init_fn=_init_fn)
            trgtrain_dataset = MRDataset(data_pth=train_mr_data_pth, gt_pth=train_mr_gt_pth,
                                         img_mean=img_mean, transform=transforms)
            trgtrain_loader = data.DataLoader(trgtrain_dataset,
                                              batch_size=cfg.TRAIN.BATCH_SIZE,
                                              num_workers=cfg.NUM_WORKERS,
                                              shuffle=True,
                                              pin_memory=True,
                                              worker_init_fn=_init_fn)
            sval_dataset = CTDataset(data_pth=val_ct_data_pth, gt_pth=val_ct_gt_pth,
                                     img_mean=img_mean, transform=transforms)
            sval_loader = data.DataLoader(sval_dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE,
                                          num_workers=cfg.NUM_WORKERS,
                                          shuffle=True,
                                          pin_memory=True,
                                          worker_init_fn=_init_fn)

        print('dataloader finish')
        print(f"Epoch {epoch} - Stage {curriculum_stage}")

        # UDA TRAINING
        train_domain_adaptation(model,strain_loader,trgtrain_loader,sval_loader,cfg,epoch)
        
        with open(os.path.join(cfg.TRAIN.SNAPSHOT_DIR, "curriculum_stage.txt"), "a") as f:
            f.write(f"Epoch {epoch}: Stage {curriculum_stage}: ")
        
        if curriculum_stage < 4:
            avg_entropy = avg_pseudo_entropy(model, trgtrain_loader, cfg)
            print(f"Avg Pseudo Entropy: {avg_entropy:.4f}")
            if avg_entropy < entropy_thresholds[curriculum_stage]:
                curriculum_stage += 1
                print(f"Switching to stage {curriculum_stage}")
                
        with open(os.path.join(cfg.TRAIN.SNAPSHOT_DIR, "curriculum_stage.txt"), "a") as f:
            f.write(f"Avg Pseudo Entropy {avg_entropy:.4f}\n")
            

if __name__ == '__main__':
    main()
