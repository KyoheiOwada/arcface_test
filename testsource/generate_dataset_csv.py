import pandas as pd
import numpy as np
import glob

def generate_dataset_csv():
  labels = []
  all_image_paths = []

  dir_names = glob.glob('/face_recognition/facescrub_images/**/', recursive=True)
  for i,dir_name in enumerate(dir_names):
    image_paths = glob.glob(dir_name + '/*.png')
    all_image_paths.extend(image_paths)
    for _ in image_paths:
      labels.append(i)
      
  print('label length{}\t path length{}'.format(len(labels),len(all_image_paths)))
  my_dict = {'label': labels, 'image_path': all_image_paths}
  my_df = pd.DataFrame.from_dict(my_dict)
  my_df.to_csv('faces_database.csv')

if __name__ == '__main__':
  generate_dataset_csv()
