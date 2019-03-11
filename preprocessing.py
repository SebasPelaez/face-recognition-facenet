import cv2
import json
import os
import shutil
import sys
import tarfile
import wget

import numpy as np
import pandas as pd

import utils

def download_data(params):
  
  url_tar_file = params['url_dataset']
  url_meta_file = params['url_metadata']
  tar_file = os.path.join(params['data_dir'],params['file_list'][0])
  metadata_file = os.path.join(params['data_dir'],params['file_list'][1])

  wget.download(url_tar_file, params['data_dir'])
  wget.download(url_meta_file, params['data_dir'])

def extract_data(params):
    
  tar_file = os.path.join(params['data_dir'],params['file_list'][0])
  
  tar = tarfile.open(tar_file, 'r:gz')
  for item in tar:
    tar.extract(item,params['data_dir'])

def generate_label_json(params):

  labels_json_path = os.path.join(params['data_dir'],params['labels_json'])

  if not os.path.isfile(labels_json_path):
    labels_dict = dict()
    for numeric_label,string_label in enumerate(os.listdir(params['data_dir_lfw'])):
      labels_dict[string_label] = numeric_label

    with open(labels_json_path,'w') as file:
      json.dump(labels_dict,file)

def complete_data(params):
  data_path = os.path.join(params['data_dir'],params['file_list'][1])
  all_data = pd.read_csv(data_path, sep="\t", header=None)

  for index, row in all_data.iterrows():

    folder_name, num_items = row

    folder_path = os.path.join(params['data_dir_lfw'],folder_name)
    image = os.listdir(folder_path)[0]
    image_path = os.path.join(folder_path,image)

    upload_image = cv2.imread(image_path,cv2.IMREAD_COLOR)

    if (num_items%4) == 1:

      flip_image = np.flip(upload_image,0)
      save_complete_image(folder_path=folder_path,folder_name=folder_name,image_id='1',flip_image=flip_image)

      flip_image = np.flip(upload_image,1)
      save_complete_image(folder_path=folder_path,folder_name=folder_name,image_id='2',flip_image=flip_image)

      flip_image = np.flip(upload_image,0)
      flip_image = np.flip(flip_image,1)
      save_complete_image(folder_path=folder_path,folder_name=folder_name,image_id='3',flip_image=flip_image)

    if (num_items%4) == 2:

      flip_image = np.flip(upload_image,0)
      save_complete_image(folder_path=folder_path,folder_name=folder_name,image_id='1',flip_image=flip_image)

      flip_image = np.flip(upload_image,1)
      save_complete_image(folder_path=folder_path,folder_name=folder_name,image_id='2',flip_image=flip_image)

    if (num_items%4) == 3:

      flip_image = np.flip(upload_image,0)
      save_complete_image(folder_path=folder_path,folder_name=folder_name,image_id='1',flip_image=flip_image)

def save_complete_image(folder_path,folder_name,image_id,flip_image):

  image_name = folder_name+'_000C'+image_id+'.jpg'

  image_save_path = os.path.join(folder_path,image_name)
  cv2.imwrite(image_save_path,flip_image)

def generate_datasets(params):
    
  data_path = os.path.join(params['data_dir'],params['file_list'][1])
  all_data = pd.read_csv(data_path, sep="\t", header=None)
  
  all_data = all_data.sample(frac=1).reset_index(drop=True)
  
  training = all_data.sample(frac = 0.7)
  validation = all_data.drop(training.index).sample(frac = 0.6) 
  test = all_data.drop(training.index).drop(validation.index)
  
  training.to_csv(os.path.join(params['data_dir'],'training.txt'), header=None, index=None, sep='\t')
  validation.to_csv(os.path.join(params['data_dir'],'validation.txt'), header=None, index=None, sep='\t')
  test.to_csv(os.path.join(params['data_dir'],'test.txt'), header=None, index=None, sep='\t')

if __name__ == '__main__':

  params = utils.yaml_to_dict('config.yml')

  file_list = params['file_list']
  dir_path = os.path.join(params['data_dir'], params['dataset_name'])

  for file_name in file_list:
    file_path = os.path.join(params['data_dir'], file_name)
    print('as',file_path)
    if os.path.exists(file_path):
      os.remove(file_path)

  if os.path.exists(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)
  
  download_data(params)
  extract_data(params)
  generate_label_json(params)
  complete_data(params)
  generate_datasets(params)


  