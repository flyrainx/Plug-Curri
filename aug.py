import os
import torch
import random
import numpy as np
from torchvision.transforms import functional as TF
from scipy.ndimage import map_coordinates, gaussian_filter


def elastic_transform(img, label, alpha=10, sigma=4, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    img_np = img.numpy()
    label_np = label.numpy()

    shape = label_np.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    img_deformed = np.zeros_like(img_np)
    for c in range(img_np.shape[0]):
        img_deformed[c] = map_coordinates(img_np[c], indices, order=1, mode='reflect').reshape(shape)

    label_deformed = map_coordinates(label_np, indices, order=0, mode='reflect').reshape(shape)

    img_tensor = torch.from_numpy(img_deformed)
    label_tensor = torch.from_numpy(label_deformed).long()

    return img_tensor, label_tensor


def random_affine(img, label, degrees=10, translate=0.05, scale=(0.9, 1.1), shear=5):
  
    angle = random.uniform(-degrees, degrees)
    trans_x = random.uniform(-translate, translate)
    trans_y = random.uniform(-translate, translate)
    scale_factor = random.uniform(scale[0], scale[1])
    shear_angle = random.uniform(-shear, shear)

    img = TF.affine(img, angle=angle, translate=[trans_x * img.shape[2], trans_y * img.shape[1]],
                    scale=scale_factor, shear=[shear_angle],
                    interpolation=TF.InterpolationMode.BILINEAR)

    label = TF.affine(label.unsqueeze(0).float(), angle=angle, translate=[trans_x * img.shape[2], trans_y * img.shape[1]],
                      scale=scale_factor, shear=[shear_angle],
                      interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()

    return img, label


def normalize(img, mean=0.5, std=0.5):
    return TF.normalize(img, [mean], [std])


def transform(img, label=None, apply_elastic=True, apply_affine=True, normalize_img=True):

    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
    if label is not None and not isinstance(label, torch.Tensor):
        label = torch.from_numpy(label)

    img = img.float()
    if label is not None:
        label = label.long()

    # Elastic deformation
    if apply_elastic and random.random() < 0.3:
        img, label = elastic_transform(img, label)

    # Affine transform
    if apply_affine and random.random() < 0.5:
        img, label = random_affine(img, label)

    # Normalize
    if normalize_img:
        img = normalize(img)

    return img, label
