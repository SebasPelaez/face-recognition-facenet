import json
import os

import pandas as pd
import tensorflow as tf

def _sources(params, mode='training'):

  file_path = mode + '.txt'
  data_path = os.path.join(params['data_dir'],file_path)
  df = pd.read_csv(data_path, sep="\t", header=None)

  if params['shuffle']:
    df = df.sample(frac=1).reset_index(drop=True)

  data = list()
  for class_sub_folder in df[0]:
    images_sub_folder = os.path.join(params['data_dir_lfw'],class_sub_folder)
    images = os.listdir(images_sub_folder)
    for image_name in images:
      image_path = os.path.join(images_sub_folder,image_name)
      data.append(image_path)

  label = _get_labels_from_json(params,data)

  return data,label

def _get_labels_from_json(params,data):

  label_json_path = os.path.join(params['data_dir'], params['labels_json'])
  with open(label_json_path, 'r') as file:
    label_dict = json.load(file)

  labels = list()
  for image_path in data:
    split_image_path = os.path.split(image_path)[0]
    name = os.path.split(split_image_path)[1]
    labels.append(label_dict[name])

  return labels

def input_fn(sources, train, params):

  def parse_image(filename,label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_decoded.set_shape((None, None, 3))
    return image_decoded,label

  image_list, labels = sources

  image_data_set = tf.data.Dataset.from_tensor_slices(image_list)
  label_data_set = tf.data.Dataset.from_tensor_slices(labels)
  
  data_set = tf.data.Dataset.zip((image_data_set,label_data_set))

  data_set = data_set.map(parse_image, num_parallel_calls=4)

  if train:
    data_set = data_set.repeat()

  data_set = data_set.batch(params['batch_size'])
  iterator = data_set.make_one_shot_iterator()

  images_batch, labels_batch = iterator.get_next()

  features = {'image': images_batch}
  y = labels_batch
  
  return features, y