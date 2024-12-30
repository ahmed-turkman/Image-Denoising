import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# creating a custom dataset class
class NoiseImagesDataset(Dataset):
  def __init__(self, img_dir):
    self.img_dir = img_dir
    self.noisy_images = os.listdir(img_dir)

  def __len__(self):
    return len(self.noisy_images)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.noisy_images[idx])
    img = Image.open(img_path)
    img = np.array(img)
    # Normalizing the image
    img = (img - img.min()) / (img.max() - img.min())
    img = img.transpose((2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32)
    # Adding a noise to the image
    noisy_img = img + torch.normal(0, 0.1 * img.max(), img.shape)
    noisy_img = (noisy_img - noisy_img.min()) / (noisy_img.max() - noisy_img.min())

    return noisy_img, img

