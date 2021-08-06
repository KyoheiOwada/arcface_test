import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model import ArcMarginProduct as Arcface

class Net(nn.Module):
  def __init__(self, model_name, use_pretrained=True):
    super(Net, self).__init__()
    if model_name == "wide_resnet101":
      self.resnet = models.wide_resnet101_2(pretrained=use_pretrained)
    elif model_name == "wide_resnet50":
      self.resnet = models.wide_resnet101_2(pretrained=use_pretrained)
    elif model_name == "resnext50":
      self.resnet = models.resnext_50_32x4d(pretrained=use_pretrained)
    elif model_name == "resnet50":
      self.resnet = models.resnet50(pretrained=use_pretrained)
    #self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    print(self.resnet.fc)
    self.resnet.fc = nn.Linear(2048, 1024)
    self.ArcFace_layer = Arcface.ArcMarginProduct(1024, 2360, easy_margin=True)
    #print(self.resnet)

  def forward(self, x, y):
    #x = x.view(-1, 25*25*128)
    x = self.resnet(x)
    x = self.ArcFace_layer(x, y)

    return x

class TrainedNet(nn.Module):
  def __init__(self, model_name, use_pretrained=True):
    super(TrainedNet, self).__init__()
    if model_name == "wide_resnet101":
      self.resnet = models.wide_resnet101_2(pretrained=use_pretrained)
    elif model_name == "wide_resnet50":
      self.resnet = models.wide_resnet101_2(pretrained=use_pretrained)
    elif model_name == "resnext50":
      self.resnet = models.resnext_50_32x4d(pretrained=use_pretrained)
    elif model_name == "resnet50":
      self.resnet = models.resnet50(pretrained=use_pretrained)
    #self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    print(self.resnet.fc)
    self.resnet.fc = nn.Linear(2048, 1024)
    self.ArcFace_layer = Arcface.ArcMarginProduct(1024, 2360, easy_margin=True)
    #print(self.resnet)

  def forward(self, x, y):
    #x = x.view(-1, 25*25*128)
    x = self.resnet(x)
    fv = x
    x = self.ArcFace_layer(x, y)

    return fv, x


