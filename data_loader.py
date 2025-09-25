from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import glob

from aug import transform

class CTDataset(Dataset):
    def __init__(self,data_pth,gt_pth,img_mean,transform_fn=None,curriculum_stage=None):

        with open(data_pth, 'r') as fp:
            self.ct_image_list = fp.readlines()

        with open(gt_pth, 'r') as fp:
            self.ct_gt_list = fp.readlines()
              
        self.transform        = transform
        self.img_mean         = img_mean
        self.curriculum_stage = curriculum_stage

        if curriculum_stage == 0:
            half_len = len(self.ct_image_list) // 2
            self.ct_image_list = self.ct_image_list[:half_len]
            self.ct_gt_list    = self.ct_gt_list[:half_len]


    def __getitem__(self, index):
        orig_len = len(self.ct_image_list)

        if self.curriculum_stage is not None and self.curriculum_stage > 1 and index >= orig_len:
            real_idx = index - orig_len
            img_pth  = self.ct_image_list[real_idx].strip()
            gt_pth   = self.ct_gt_list[real_idx].strip()
            img, gt  = self.load_data(img_pth, gt_pth)
            img, gt  = self.transform_fn(img, gt)
        else:
            img_pth  = self.ct_image_list[index % orig_len].strip()
            gt_pth   = self.ct_gt_list[index % orig_len].strip()
            img, gt  = self.load_data(img_pth, gt_pth)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        gt  = torch.from_numpy(gt.astype(int)).long()
        
        return img, gt, index

    
    def __len__(self):
        if self.curriculum_stage is not None and self.curriculum_stage > 1:
            return len(self.ct_image_list) * 2
        return len(self.ct_image_list)

          
    def load_data(self,img_pth, gt_pth):
        img = np.load(img_pth) # h*w*1
        gt  = np.load(gt_pth)  # h*w

        img = np.expand_dims(img,-1)
        img = np.tile(img,[1,1,3])  # h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img, gt


class MRDataset(Dataset):
    def __init__(self, data_pth, gt_pth,img_mean, transform_fn=None,curriculum_stage=None):
        with open(data_pth, 'r') as fp:
            self.mr_image_list = fp.readlines()

        with open(gt_pth, 'r') as fp:
            self.mr_gt_list = fp.readlines()
              
        self.transform        = transform
        self.img_mean         = img_mean
        self.curriculum_stage = curriculum_stage

              
    def __getitem__(self, index):
        orig_len = len(self.mr_image_list)

        if self.curriculum_stage is not None and self.curriculum_stage > 1 and index >= orig_len:
            real_idx = index - orig_len
            img_pth  = self.mr_image_list[real_idx].strip()
            gt_pth   = self.mr_gt_list[real_idx].strip()
            img, gt  = self.load_data(img_pth, gt_pth)
            img, gt  = self.transform_fn(img, gt)
        else:
            img_pth  = self.mr_image_list[index % orig_len].strip()
            gt_pth   = self.mr_gt_list[index % orig_len].strip()
            img, gt  = self.load_data(img_pth, gt_pth)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        gt  = torch.from_numpy(gt.astype(int)).long()
        
        return img, gt, index

    
    def __len__(self):
        if self.curriculum_stage is not None and self.curriculum_stage > 1:
            return len(self.mr_image_list) * 2
        return len(self.mr_image_list)

    
    def load_data(self, img_pth, gt_pth):
        img = np.load(img_pth)  # h*w*1
        gt  = np.load(gt_pth)
        img = np.expand_dims(img,-1)
        img = np.tile(img,[1,1,3])  # h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img, gt
