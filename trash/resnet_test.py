import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms

resnet = models.resnet18(pretrained=True)
print(resnet)
print("use resnet architecture")
#for i in resnet.fc:
#  print(i)

model = models.resnet18(pretrained=True)
#newmodel = torch.nn.Sequential(*(list(model.children())[:-2]))
print(model)
"""for param in model.parameters():
  param.requires_grad = False
  print(param)"""
"""params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

update_param_names_1 = ["features"]
update_param_names_2 = ["classifier.0.weight","classifier.0.bias","classifier.3.weight","classifier.3.bias"]
update_param_names_3 = ["classifier.6.weight","classifier.6.bias"]

for name, param in model.named_parameters():
  if update_param_names_1[0] in name:
    param.requires_grad = True
    params_to_update_1.append(param)
    print("params_to_update_1 loaded", name)

  elif name in update_param_names_2:
    param.requires_grad = True
    params_to_update_2.append(param)
    print("params_to_update_2 loaded.:", name)

  elif name in update_param_names_3:
    param.requires_grad = True
    params_to_update_3.append(param)
    print("params_to_update_3 loaded.:", name)

  else:
    param.requires_grad = False
    print("no grad computing", name)"""
