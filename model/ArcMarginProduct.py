import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

class ArcMarginProduct(nn.Module):
  def __init__(self, in_features, out_features, s=30.0, m=0.60, easy_margin=False):
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

