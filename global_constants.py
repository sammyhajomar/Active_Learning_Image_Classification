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
  LABELED_DIR = "Dataset/Labeled"
  EVAL_DIR = "Dataset/Eval"
  UNLABELED_DIR = "Dataset/Unlabeled"
  UNSURE_DIR = 'Dataset/Labeled/Unsure'
  SWIPE_LABELLER_DIR = "external_lib/Swipe-Labeler/api/api.py"
  TBL_DIR = "Dataset/To_Be_Labeled"
  IMAGE_PATH_COL = 'image_paths'

else:
  LABELED_DIR = f"{dataset_name}/Labeled"
  EVAL_DIR = f"{dataset_name}/Eval"
  UNLABELED_DIR = f"{dataset_name}/Unlabeled"
  UNSURE_DIR = f'{dataset_name}/Labeled/Unsure'
  SWIPE_LABELLER_DIR = "external_lib/Swipe-Labeler/api/api.py"
  TBL_DIR = f"{dataset_name}/To_Be_Labeled"
  VALID_DIR = f"{dataset_name}/Validation"

  IMAGE_PATH_COL = 'image_paths'