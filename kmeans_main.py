import glob
from opencv_face_detection import detect_face, Dataset
from utils import get_features_func
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import cos_sim
import cv2
import numpy as np
from sklearn import preprocessing, cluster
from spherecluster import SphericalKMeans
from spherecluster import VonMisesFisherMixture
from PIL import Image

if __name__ == '__main__':
  frame_path_list = glob.glob('./opencv_face_detection/frame/*.jpg')
  feature_vector_list = []
  cos_sim_list = []
  rect_face_imgs_list = []
  pil_list = []
  face_id = 0
  threshold = 0.3
  mean = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)
  print(len(frame_path_list))

  for frame in frame_path_list:
    print('cliping images... \n')
    img = cv2.imread(frame)
    rect_face_imgs = detect_face.detect_face(img)
    rect_face_imgs_list.append(rect_face_imgs)
    for img_array in rect_face_imgs:
      rect_face_imgs_pil = Image.fromarray(img_array)
      pil_list.append(rect_face_imgs_pil)
    #print('rect face imgs list shape is:', pil_list)
    print('loading dataset \n')

  dataset = Dataset.MyDataset(pil_list,
                                    transform=transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean, std)
                                                                 ]))
    #dataset = Dataset.MyDataset(rect_face_imgs)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
  #for (image, label) in dataloader:
    #print('extracting feature...\n')
    #feature_vector = get_features_func.get_features_func(image, label)
    #feature_vector_list.append(feature_vector)
   #cos_sim_list = []
  #for vec in feature_vector_list:
    #print(feature_vector.reshape(-1, 1).shape, vec.shape)
      #sim = cos_sim.cos_sim(feature_vector, vec.reshape(-1, 1))
      #cos_sim_list.append(sim)
  #feature_vector_list = np.array(feature_vector_list)
  #feature_vector_list = feature_vector_list.reshape(dataset.__len__(), -1)
  #print('save feature list\n', feature_vector_list)
  #np.save('./feature_list', feature_vector_list)
  feature_vector_list = np.load('./feature_list.npy')
  #normalized_vec_list = preprocessing.normalize(feature_vector_list)
  #kmeans_model = cluster.KMeans(n_clusters=20, random_state=10).fit(normalized_vec_list)
  vmf = VonMisesFisherMixture(n_clusters=10, posterior_type='hard').fit(feature_vector_list)
  labels = vmf.labels_

  for (image, _), label in zip(dataloader, labels):
    new_image = np.array(image[0])
    new_image = new_image.transpose(1,2,0)
    print('image shape is:', new_image.shape)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    cv2.imshow(str(label), new_image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
  for label in labels:
    print(label)
