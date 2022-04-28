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


class PrepareData:
    def __init__(self,dataset_name,positive_class):
        self.positive_class = positive_class
        self.dataset_name = dataset_name

    def tfds_io(self,ds,set_path,val=False):
      img_paths = []
      img_num = 0
      batch_bar = tqdm(total=len(ds), dynamic_ncols=True, leave=False, position=0, desc='Download and save dataset') 
      for img,label in ds:
          label = label.numpy()
          serialized_im = tf.image.encode_jpeg(img)
          img_path = f'{label}_{img_num}.jpg'
          img_paths.append(img_path)
          if val:
            if img_path.split('/')[-1].split('_')[0] == self.positive_class:
              path_and_name = os.path.join(self.dataset_name,set_path,"positive",img_path)
            else:
              path_and_name = os.path.join(self.dataset_name,set_path,"negative",img_path)
          else:
            path_and_name = os.path.join(self.dataset_name,set_path,img_path)
          tf.io.write_file(path_and_name, serialized_im)
          img_num += 1
          batch_bar.update()
      batch_bar.close()


    def download_and_prepare(self):
        tfds.disable_progress_bar()

        try:
          ds = tfds.load(
              self.dataset_name,
              split=['train','test'],
              shuffle_files = True,
              as_supervised=True)
              
          if not os.path.exists(self.dataset_name):
            self.tfds_io(ds[0],"Unlabeled")
            self.tfds_io(ds[1],"Validation",val=True)


        except ValueError:
          ds = tfds.load(
              self.dataset_name,
              split='train',
              shuffle_files = True,
              as_supervised=True)
          
          if not os.path.exists(self.dataset_name):
            self.tfds_io(ds[0],"Unlabeled", self.dataset_name)


        if not os.path.exists(f'{self.dataset_name}/Eval'):
          os.makedirs(f'{self.dataset_name}/Eval')
          os.makedirs(f'{self.dataset_name}/Eval/positive')
          os.makedirs(f'{self.dataset_name}/Eval/negative')

        if not os.path.exists(f'{self.dataset_name}/Labeled'): 
          os.makedirs(f'{self.dataset_name}/Labeled/')
          os.makedirs(f'{self.dataset_name}/Labeled/positive')
          os.makedirs(f'{self.dataset_name}/Labeled/negative')

        if not os.path.exists(f'{self.dataset_name}/Validation'):  
          os.makedirs(f'{self.dataset_name}/Validation/positive')
          os.makedirs(f'{self.dataset_name}/Validation/negative')
        
        if os.path.exists('checkpoints/'):
          shutil.rmtree('checkpoints/')
        os.makedirs('checkpoints/')

        print(f'Downloaded and prepared {self.dataset_name}')


def tfds_annotate(image_paths, num_images, already_labeled, positive_class, labeled_dir, val = False):
  if not val:
    num_labeled = 0
    positive_labels = 0
    negative_labels = 0
    shuffle(image_paths)
    for image in image_paths:
      if image not in already_labeled:
        if image.split('/')[-1].split('_')[0] == positive_class and positive_labels != int(num_images / 2):
          shutil.copy(image, os.path.join(labeled_dir,'positive',image.split('/')[-1]))
          positive_labels += 1
          num_labeled += 1
          already_labeled.append(image)
        elif negative_labels != int(num_images / 2):
          shutil.copy(image, os.path.join(labeled_dir,'negative',image.split('/')[-1]))
          negative_labels += 1
          num_labeled += 1
          already_labeled.append(image)
      if num_labeled==num_images:
        break
    return already_labeled

  else:
    num_labeled = 0
    num_images_pos = num_images // 2
    all_pos_images = [image for image in image_paths if image.split('/')[-1].split('_')[0] == positive_class]
    all_neg_images = [image for image in image_paths if not image.split('/')[-1].split('_')[0] == positive_class]
    shuffle(all_pos_images)
    pos_images = [all_pos_images[i] for i in range(num_images_pos)]
    neg_images = [all_neg_images[i] for i in range(num_images_pos)]
        
    for image in pos_images + neg_images:
      if image not in already_labeled:
        num_labeled += 1
        already_labeled.append(image)
        if image.split('/')[-1].split('_')[0] == positive_class:
          shutil.copy(image, os.path.join(labeled_dir,'positive',image.split('/')[-1]))
        else:
          shutil.copy(image, os.path.join(labeled_dir,'negative',image.split('/')[-1]))
      if num_labeled==num_images:
        break
    return already_labeled





