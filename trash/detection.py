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
import my_dataset
from torch.utils.data import Dataset
import numpy as np
from model import model
from utils import cos_sim

if __name__=='__main__':
  db_file_path = 'utils/detection_val.csv'
  pred_file_path = 'utils/detection_db.csv'
  ROOT_DIR = ""
  mini_batch = 1
  out_feature = 1024
  mean = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)

  database = my_dataset.MyDataset(db_file_path,
                                    transform=transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)]))
  pred = my_dataset.MyDataset(pred_file_path,
                                    transform=transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)]))
  pred_loader = torch.utils.data.DataLoader(pred, batch_size=mini_batch, shuffle=False,num_workers=0, pin_memory=True)
  GPU = True
  device = torch.device("cuda" if GPU else "cpu")

  model = model.TrainedNet("wide_resnet101")
  model.load_state_dict(
          torch.load('model_weight/wideresnet101_weight.pth'))

  #print(model)
  model = model.to(device)
  model.eval()

  #predict similarity of faces by using cos_sim
  feature_vectors_list = np.load('./feature_vectors.npy')
  correct = 0
  for (image, label) in pred_loader:
    image, label = Variable(image.float(), volatile=True).to(device), Variable(label).to(device)
    pred_feature_vector, _ = model(image, label)
    pred_feature_vector = pred_feature_vector.data.cpu().numpy()
    
    cos_sim_list = []

    for feature_vector in feature_vectors_list:
      cos_sim_list.append(cos_sim.cos_sim(pred_feature_vector, feature_vector))
    print([len(v) for v in cos_sim_list])
    pred_index = cos_sim_list.index(max(cos_sim_list))
    print("predicted label is: {}".format(database.__getitem__(pred_index)[1]))
    print("correct label is: {}".format(label[0]))

    if label[0] == database.__getitem__(pred_index)[1]:
      correct += 1
  print("accuracy: {}/{} ({:.0f}%)\n".format(correct, pred.__len__(), 100. * correct / pred.__len__()))

