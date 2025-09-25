from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import glob

class CTDataset(Dataset):
    def __init__(self,data_pth,gt_pth,img_mean,transform=None,curriculum_stage=None):

        with open(data_pth, 'r') as fp:
            self.ct_image_list = fp.readlines()

        with open(gt_pth, 'r') as fp:
            self.ct_gt_list = fp.readlines()
              
        self.transform        = transform
        self.img_mean         = img_mean
        self.curriculum_stage = curriculum_stage


    def __getitem__(self, index):

        img_pth = self.ct_image_list[index][:-1]
        gt_pth  = self.ct_gt_list[index][:-1]
        img,gt  = self.load_data(img_pth,gt_pth)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.transpose(img, (2, 0, 1))  # 3*h*w

        gt = gt.astype(int)

        return img, gt,index

          
    def __len__(self):

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

      
    def gl_load_data(self,img_pth):
        img = np.load(img_pth)  # h*w*1
        img = np.expand_dims(img,-1)
        img = np.tile(img, [1, 1, 3])  # h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img


class MRDataset(Dataset):
    def __init__(self, data_pth, gt_pth,img_mean, transform=None,curriculum_stage=None):
        with open(data_pth, 'r') as fp:
            self.mr_image_list = fp.readlines()

        with open(gt_pth, 'r') as fp:
            self.mr_gt_list = fp.readlines()
              
        self.transform        = transform
        self.img_mean         = img_mean
        self.curriculum_stage = curriculum_stage

              
    def __getitem__(self, index):

        img_pth = self.mr_image_list[index][:-1]
        gt_pth = self.mr_gt_list[index][:-1]
        img, gt = self.load_data(img_pth, gt_pth)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.transpose(img, (2, 0, 1))  # 3*h*w

        gt = gt.astype(int)
        return img, gt, index

    def __len__(self):
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

    def gl_load_data(self,img_pth):
        img = np.load(img_pth)  # h*w*1
        img = np.expand_dims(img,-1)
        img = np.tile(img, [1, 1, 3])  # h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img
