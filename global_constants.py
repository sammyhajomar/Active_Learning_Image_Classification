import os
from utils import load_config
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to config file")
args = parser.parse_args()
config_path = args.config_path                      

config = load_config(config_path)
dataset_name = config['data']['dataset_name']

if dataset_name == 'resisc':
  LABELLED_DIR = "Dataset/Labelled"
  EVAL_DIR = "Dataset/Eval"
  UNLABELLED_DIR = "Dataset/Unlabelled"
  UNSURE_DIR = 'Dataset/Labelled/Unsure'
  SWIPE_LABELLER_DIR = "external_lib/Swipe-Labeler/api/api.py"
  TBL_DIR = "Dataset/To_Be_Labelled"
  IMAGE_PATH_COL = 'image_paths'

else:
  LABELLED_DIR = f"{dataset_name}/Labelled"
  EVAL_DIR = f"{dataset_name}/Eval"
  UNLABELLED_DIR = f"{dataset_name}/Unlabelled"
  UNSURE_DIR = f'{dataset_name}/Labelled/Unsure'
  SWIPE_LABELLER_DIR = "external_lib/Swipe-Labeler/api/api.py"
  TBL_DIR = f"{dataset_name}/To_Be_Labelled"

  IMAGE_PATH_COL = 'image_paths'