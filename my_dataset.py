import torch
from torch.utils.data import Dataset
import os
import numpy as np
import sys
import pandas as pd
import skimage
from torchvision import transforms
from PIL import Image

LABEL_IDX = 2
IMG_IDX = 1

class MyDataset(Dataset):

  def __init__(self, csv_file_path, transform=None):
    self.image_dataframe = pd.read_csv(csv_file_path)
    #self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.image_dataframe)

  def __getitem__(self, idx):
    label = self.image_dataframe.iat[idx, LABEL_IDX]
    img_name = self.image_dataframe.iat[idx, IMG_IDX]
    #print(img_name)
    #print(label)
    img = Image.open(img_name)
    img = img.convert('RGB')
    if self.transform:
      img = self.transform(img)
    return img, label

if __name__=='__main__':
    input_file_path = './faces.csv'
    imgDataset = MyDataset(input_file_path, transform=transforms.Compose([transforms.ToTensor()]))
    print(imgDataset.__getitem__(0)[0].size())
