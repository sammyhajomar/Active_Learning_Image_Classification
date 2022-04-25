import os
import shutil
import pathlib
from imutils import paths
from random import shuffle
import subprocess
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np

def download_and_prepare(dataset='cifar10'):
    ds,info = tfds.load(
        dataset,
        split='train',
        shuffle_files = True,
        as_supervised=True,
        with_info=True)
    
    root_path = f'/content/Active-Labeler/{dataset}'
    if not os.path.exists(dataset):
      img_paths = []
      img_num = 0
      for img,label in ds:
          label = label.numpy()
          serialized_im = tf.image.encode_jpeg(img)
          
          img_path = f'{label}_{img_num}.jpg'
          img_paths.append(img_path)
          path_and_name = os.path.join(f'{root_path}/Unlabelled', img_path)
          tf.io.write_file(path_and_name, serialized_im)
          img_num += 1

    if not os.path.exists(f'{root_path}/Eval'):
      os.makedirs(f'{root_path}/Eval')
      os.makedirs(f'{root_path}/Eval/positive')
      os.makedirs(f'{root_path}/Eval/negative')

    if not os.path.exists(f'{root_path}/Labelled'): 
      os.makedirs(f'{root_path}/Labelled/')
      os.makedirs(f'{root_path}/Labelled/positive')
      os.makedirs(f'{root_path}/Labelled/negative')

    if not os.path.exists(f'{root_path}/To_Be_Labelled'):  
      os.makedirs(f'{root_path}/To_Be_Labelled')
    
    if os.path.exists('checkpoints/'):
      shutil.rmtree('checkpoints/')
    os.makedirs('checkpoints/')
  

    # df = tfds.as_dataframe(ds, info)
    print(f'Downloaded and prepared {dataset}')


def tfds_annotate(image_paths, num_images, already_labelled, positive_class, labelled_dir="cifar10/Labeled", val = False):
  if not val:
    num_labelled = 0
    positive_labels = 0
    negative_labels = 0
    shuffle(image_paths)
    for image in image_paths:
      if image not in already_labelled:
        if image.split('/')[-1].split('_')[0] == positive_class and positive_labels != int(num_images / 2):
          shutil.copy(image, os.path.join(labelled_dir,'positive',image.split('/')[-1]))
          positive_labels += 1
          num_labelled += 1
          already_labelled.append(image)
        elif negative_labels != int(num_images / 2):
          shutil.copy(image, os.path.join(labelled_dir,'negative',image.split('/')[-1]))
          negative_labels += 1
          num_labelled += 1
          already_labelled.append(image)
      if num_labelled==num_images:
        break
    return already_labelled

  else:
    num_labelled = 0
    num_images_pos = num_images // 2
    print("positive_class",type(positive_class),positive_class)
    ["image_paths[0]",image_paths[0]]
    all_pos_images = [image for image in image_paths if image.split('/')[-1].split('_')[0] == positive_class]
    all_neg_images = [image for image in image_paths if not image.split('/')[-1].split('_')[0] == positive_class]
    shuffle(all_pos_images)
    pos_images = [all_pos_images[i] for i in range(num_images_pos)]
    neg_images = [all_neg_images[i] for i in range(num_images_pos)]
        
    for image in pos_images + neg_images:
      if image not in already_labelled:
        num_labelled += 1
        already_labelled.append(image)
        if image.split('/')[-1].split('_')[0] == positive_class:
          shutil.copy(image, os.path.join(labelled_dir,'positive',image.split('/')[-1]))
        else:
          shutil.copy(image, os.path.join(labelled_dir,'negative',image.split('/')[-1]))
      if num_labelled==num_images:
        break
    return already_labelled





