from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from torch.nn import Parameter
from torch.autograd import Variable
import torch.optim as optim
import math
from sklearn.model_selection import train_test_split
import my_dataset
import train_and_test

class ArcMarginProduct(nn.Module):
  def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
    super(ArcMarginProduct, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.s = s
    self.m = m
    self.weight = Parameter(torch.FloatTensor(out_features, in_features))
    nn.init.xavier_uniform_(self.weight) # ???あｔoで調べて

    self.easy_margin = easy_margin
    self.cos_m = math.cos(m)
    self.sin_m = math.sin(m)
    self.th = math.cos(math.pi - m)
    self.mm = math.sin(math.pi - m) * m #???

  def forward(self, input, label):
    cosine = F.linear(F.normalize(input), F.normalize(self.weight))
    sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0,1))
    phi = cosine * self.cos_m - sine * self.sin_m #加法定理でcos(th + m)を求める
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
    self.resnet = models.resnet50(pretrained=True)
    self.fc1 = nn.Linear(25*25*128, 128)
    self.ArcFace_layer = ArcMarginProduct(128,71)
    
  def forward(self, x, y):
    #x = x.view(-1, 25*25*128)
    x = self.resnet(x)
    x = self.fc1(x)
    x = self.ArcFace_layer(x, y)

    return x

if __name__ == '__main__':

  input_file_path = './faces.csv'
  imgDataset = my_dataset.MyDataset(input_file_path, transform=transforms.Compose([transforms.ToTensor()]))
  print(imgDataset.__getitem__(0)[0].size())
  
# load images
  train_data, test_data = train_test_split(imgDataset, test_size=0.2)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

  GPU = True
  device = torch.device("cuda" if GPU else "cpu")

  net = CNN()
  print("loading_the model")
# print(torch.load('model_weight.pth'))
# net.load_state_dict(torch.load('model/model_weight.pth'))
  
  print(net)
  net = net.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  net.train()
  for epoch in range(1, 10+1):
    if epoch % 100 == 0:
        torch.save(net.state_dict(), "./model/model_weight.pth")
        print('saved the model.')
    for batch_idx, (image, label) in enumerate(train_loader):
      image, label = Variable(image).cuda(0), Variable(label).cuda(0)
      optimizer.zero_grad()
      output = net(image, label)
      print(output.size())
      loss = criterion(output, label)
      loss.backward()
      optimizer.step()

      print('epoch: {}\t Loss: {}'.format(epoch , loss.data))

  test_loss = 0
  correct = 0

  net.eval()
  for (image, label) in test_loader:
    image, label = Variable(image.float(), volatile=True).cuda(0), Variable(label).cuda(0)
    output = net(image, label)
    print(output.size())
    test_loss += criterion(output, label).data
    pred = output.data.max(1, keepdim=True)[1]
    #print(pred)
    correct += pred.eq(label.data.view_as(pred)).long().cpu().sum()
  test_loss /= len(test_loader.dataset)
  print('\ntestset: average loss:{:.4f}, accuracy:{}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(image), len(train_loader.dataset),
    #                                                       100 * batch_idx / len(train_loader), loss.data[0]))

