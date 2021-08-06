from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.nn import Parameter
from torch.autograd import Variable
import torch.optim as optim
import math
from sklearn.model_selection import train_test_split
import my_dataset
from torch.utils.data import Dataset
import os
import sys
import pandas as pd
import skimage
from PIL import Image
import numpy as np
from model import model

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
    print(img_name)
    #print(label)
    img = skimage.io.imread(img_name)
    if self.transform:
      img = self.transform(img)
    return img, label


if __name__=='__main__':
  db_file_path = 'utils/detection_db.csv'
  pred_file_path = 'utils/detection_val.csv'
  ROOT_DIR = ""
  mini_batch = 100
  out_feature = 1024
  #imgDatabase = MyDataset(input_file_path, transform=transforms.Compose([
  #  transforms.ToTensor()]))

  #database, pred = train_test_split(imgDatabase, test_size=0.3)
  #print(database.__len__(), pred.__len__())
  #database_loader = torch.utils.data.DataLoader(database, batch_size=mini_batch, shuffle=False)
  #pred_loader = torch.utils.data.DataLoader(pred, batch_size=1, shuffle=False)
  database = my_dataset.MyDataset(db_file_path,
                                    transform=transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)]))
  pred = my_dataset.MyDataset(pred_file_path,
                                    transform=transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)]))
  GPU = True
  device = torch.device("cuda" if GPU else "cpu")

  model = model.Net("wide_resnet101")
  model.load_state_dict(
          torch.load('model_weight/wideresnet101_model_weight.pth'))

  #print(model)
  model = model.to(device)
  model.eval()

  feature_vectors_list = np.zeros((mini_batch,out_feature))
  #save latent codes
  for (image, label) in database:
    image, label = Variable(image.float(), volatile=True).to(device), Variable(label).to(device)
    feature_vectors, _ = model(image, label)
    print(feature_vectors.data.cpu().device)
    feature_vectors = feature_vectors.data.cpu().numpy()
    feature_vectors_list = np.concatenate((feature_vectors_list, feature_vectors))

  feature_vectors_list = np.delete(feature_vectors_list, np.s_[0:mini_batch], 0)
  np.save('./feature_vectors.npy', feature_vectors_list)

  #predict similarity of faces by using cos_sim
  correct = 0
  for (image, label) in pred:
    image, label = Variable(image.float(), volatile=True).to(device), Variable(label).to(device)
    pred_feature_vector, _ = model(image, label)
    pred_feature_vector = pred_feature_vector.data.cpu().numpy()
    
    cos_sim_list = []

    for feature_vector in feature_vectors_list:
      cos_sim_list.append(cos_sim(pred_feature_vector, feature_vector))
    pred_index = cos_sim_list.index(max(cos_sim_list))
    print("predicted label is: {}".format(database.__getitem__(pred_index)[1]))
    print("correct label is: {}".format(label[0]))

    if label[0] == database.__getitem__(pred_index)[1]:
      correct += 1
  print("accuracy: {}/{} ({:.0f}%)\n".format(correct, pred.__len__(), 100. * correct / pred.__len__()))

