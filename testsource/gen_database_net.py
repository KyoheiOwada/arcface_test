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

class ArcMarginProduct(nn.Module):
  def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
    super(ArcMarginProduct, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.s = s
    self.m = m
    self.weight = Parameter(torch.FloatTensor(out_features, in_features))
    nn.init.xavier_uniform_(self.weight) # ???�~A~B��~To�~A�調�~A��~A�

    self.easy_margin = easy_margin
    self.cos_m = math.cos(m)
    self.sin_m = math.sin(m)
    self.th = math.cos(math.pi - m)
    self.mm = math.sin(math.pi - m) * m #???

  def forward(self, input, label):
    cosine = F.linear(F.normalize(input), F.normalize(self.weight))
    sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0,1))
    phi = cosine * self.cos_m - sine * self.sin_m #�~J| ��~U��~Z�~P~F�~A�cos(th + m)�~B~R��~B�~B~A�~B~K
    if self.easy_margin:
      phi = torch.where(cosine > 0, phi, cosine)
    else:
      phi = torch.where(cosine > self.th, phi, cosine - self.mm)

    one_hot = torch.zeros(cosine.size(), device='cuda')
    one_hot.scatter_(1,label.view(-1,1).long(), 1)
    output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
    output *= self.s
    print(output)

    return output

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, 5) #input:112 output:108
    self.conv2 = nn.Conv2d(64,128, 5) #input:54 output: 50
    #self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(25*25*128, 128)
    self.ArcFace_layer = ArcMarginProduct(128,71)
    #self.fc2 = ArcFace_layer.forward(128, 81)
    #self.fc2 = nn.Linear(128, 81)

  def forward(self, x, y):
    x = F.relu(F.max_pool2d(self.conv1(x), 2,stride=(2,2)))
    x = F.relu(F.max_pool2d(self.conv2(x), 2,stride=(2,2)))
    x = x.view(-1, 25*25*128)
    fv = self.fc1(x)
    x = self.ArcFace_layer(fv, y)
    #x = ArcFace_layer(x, y)
    #x = F.log_softmax(x, dim=1)
    #x = F.softmax(x, dim=1)

    return fv, x

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__=='__main__':
  input_file_path = './faces_database.csv'
  ROOT_DIR = ""

  imgDatabase = MyDataset(input_file_path, transform=transforms.Compose([
    transforms.ToTensor()]))

  database, pred = train_test_split(imgDatabase, test_size=0.3)
  print(database.__len__(), pred.__len__())
  database_loader = torch.utils.data.DataLoader(database, batch_size=100, shuffle=False)
  pred_loader = torch.utils.data.DataLoader(pred, batch_size=1, shuffle=False)

  GPU = True
  device = torch.device("cuda" if GPU else "cpu")

  model = CNN()
  model.load_state_dict(torch.load('model/model_weight.pth'))

  #print(model)
  model = model.to(device)
  model.eval()

  feature_vectors_list = np.zeros((100,128))
  #save latent codes
  for (image, label) in database_loader:
    image, label = Variable(image.float(), volatile=True).to(device), Variable(label).to(device)
    feature_vectors, _ = model(image, label)
    print(feature_vectors.data.cpu().device)
    feature_vectors = feature_vectors.data.cpu().numpy()
    feature_vectors_list = np.concatenate((feature_vectors_list, feature_vectors))

  feature_vectors_list = np.delete(feature_vectors_list, np.s_[0:100], 0)
  np.save('./feature_vectors.npy', feature_vectors_list)

  #predict similarity of faces by using cos_sim
  correct = 0
  for (image, label) in pred_loader:
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

"""  for (image, label) in test_loader:
    image, label = Variable(image.float(), volatile=True).cuda(0), Variable(label).cuda(0)
    output = net(image, label)
    print(output.size())
    test_loss += criterion(output, label).data
    pred = output.data.max(1, keepdim=True)[1]
    #print(pred)
    correct += pred.eq(label.data.view_as(pred)).long().cpu().sum()
  test_loss /= len(test_loader.dataset)
  print('\ntestset: average loss:{:.4f}, accuracy:{}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
"""
