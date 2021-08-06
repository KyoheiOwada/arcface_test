from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.nn import Parameter
from torch.autograd import Variable
import torch.optim as optim
from sklearn.model_selection import train_test_split
import my_dataset
from torch.utils.data import Dataset
import numpy as np
from model import model

if __name__=='__main__':
  db_file_path = 'utils/detection_db.csv'
  pred_file_path = 'utils/detection_val.csv'
  ROOT_DIR = ""
  mini_batch = 1
  out_feature = 1024
  mean = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)

  database = my_dataset.MyDataset(db_file_path,
                                    transform=transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)]))
  database_loader = torch.utils.data.DataLoader(database, batch_size=mini_batch, shuffle=False,num_workers=0, pin_memory=True)
  GPU = True
  device = torch.device("cuda" if GPU else "cpu")

  model = model.TrainedNet("wide_resnet101")
  model.load_state_dict(
          torch.load('model_weight/wideresnet101_weight.pth'))

  #print(model)
  model = model.to(device)
  model.eval()
  i = 0
  f_c = 0
  sum_vector = np.zeros((1, out_feature))
  feature_vectors_list = np.zeros((mini_batch,out_feature))
  mean_vector = np.zeros((1, out_feature))

  #save latent codes
  for (image, label) in database_loader:
    print(int(label))
    if i == int(label):
      image, label = Variable(image.float(), volatile=True).cuda(0), Variable(label).cuda(0)
      feature_vectors, _ = model(image, label)
      print(feature_vectors.data.cpu().device)
      feature_vectors = feature_vectors.data.cpu().numpy()
      sum_vector = sum_vector + feature_vectors
      print(sum_vector.shape)
      f_c += 1
      #feature_vectors_list = np.concatenate((feature_vectors_list, feature_vectors))
    else:
      print("incriment i")
      i += 1
      mean_vector = sum_vector / f_c
      sum_vector = np.zeros((1, out_feature))
      sum_vector = sum_vector + feature_vectors
      f_c = 1
      feature_vectors_list = np.concatenate((feature_vectors_list, mean_vector))
  mean_vector = sum_vector / f_c
  feature_vectors_list = np.concatenate((feature_vectors_list, mean_vector))
  feature_vectors_list = np.delete(feature_vectors_list, np.s_[0:mini_batch], 0)
  print(feature_vectors_list.shape)
  np.save('./feature_vectors.npy', feature_vectors_list)

