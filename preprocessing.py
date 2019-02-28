import json
import os

def generate_label_json(params):

  labels_json_path = os.path.join(params['data_dir'],params['labels_json'])

  if not os.path.isfile(labels_json_path):
    labels_dict = dict()
    for numeric_label,string_label in enumerate(os.listdir(params['data_dir_lfw'])):
      labels_dict[string_label] = numeric_label

    with open(labels_json_path,'w') as file:
      json.dump(labels_dict,file)