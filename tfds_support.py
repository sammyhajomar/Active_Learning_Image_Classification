import os
import random
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
import gdown


class PrepareData:
    def __init__(self,dataset_name,config):
        self.dataset_name = dataset_name
        self.config = config

    def tfds_io(self,ds):
      img_paths = []
      img_num = 0
      set_type = {"train": 0, "valid": 0, "test": 0}
      batch_bar = tqdm(total=len(ds), dynamic_ncols=True, leave=False, position=0, desc='Download and save dataset') 
      for img,label in ds:
          label = label.numpy()
          serialized_im = tf.image.encode_jpeg(img)
          img_path = f'{label}_{img_num}.jpg'
          if set_type['train'] < len(ds)*0.8:
            path_and_name = os.path.join(self.dataset_name,"unlabeled",img_path)
            set_type['train'] += 1
          elif set_type['valid'] < len(ds)*0.10:
            class_idx = img_path.split('/')[-1].split('_')[0]
            path_and_name = os.path.join(self.dataset_name,"validation",class_idx,img_path)  
            set_type['valid'] += 1
          else:
            class_idx = img_path.split('/')[-1].split('_')[0]
            path_and_name = os.path.join(self.dataset_name,"test",class_idx,img_path)
            set_type['test'] += 1

          tf.io.write_file(path_and_name, serialized_im)
          img_num += 1
          img_paths.append(img_path)
          batch_bar.update()
      batch_bar.close()
      print("set_type",set_type)


    def download_and_prepare(self):
      if self.config['data']['dataset_name'] != 'resisc45':
        tfds.disable_progress_bar()

        try:
          ds,info = tfds.load(
              self.dataset_name,
              split='train+test',
              shuffle_files=True,
              as_supervised=True,
              with_info=True)
              
        except ValueError:
          ds,info = tfds.load(
              self.dataset_name,
              split='train',
              shuffle_files=True,
              as_supervised=True,
              with_info=True)
          
        if not os.path.exists(self.dataset_name):
          self.tfds_io(ds)

        if not os.path.exists(f'{self.dataset_name}/labeled'): 
          os.makedirs(f'{self.dataset_name}/labeled/')
          for i in range(info.features['label'].num_classes):
            os.makedirs(f'{self.dataset_name}/labeled/{i}')
        
        if os.path.exists('checkpoints/'):
          shutil.rmtree('checkpoints/')
        os.makedirs('checkpoints/')
    
      else:
        full_path = os.path.join(os.getcwd(), "RESISC45")
        # if os.path.exists(full_path):
        #     shutil.rmtree(full_path)
        download_path = os.path.join(os.getcwd(), "RESISC45.rar")
        url = 'https://drive.google.com/uc?id=14zEhqi9mczZaLEb33TQuKbhmurn2ClGL&export=download'
        output = 'RESISC45.rar'
        if not os.path.isfile(download_path):
          gdown.download(url, output, quiet=False)

        if not os.path.exists('RESISC45'):
          commands = [
          "unrar x {}".format(download_path),
          "mv NWPU-RESISC45/ RESISC45/"]
          # "rm {}".format(download_path)]
          os.system( " && ".join(commands))

        folder = 'Dataset/Unlabelled'
        if os.path.exists(folder):
            shutil.rmtree(folder)

        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        for i in tqdm(paths.list_images(full_path)):
            shutil.copy(i, os.path.join(folder, i.split('/')[-1]))
        print('Downloaded and prepared RESISC-45')

        print(f'Downloaded and prepared {self.dataset_name}')


def tfds_annotate(image_paths, num_images, already_labeled, labeled_dir, val=False):
  # if not val:
    num_labeled = 0
    shuffle(image_paths)
    label_types = {}
    for img in image_paths:
      label = img.split('/')[-1].split('_')[0]
      if label not in label_types:
        label_types[label] = 1
        shutil.copy(img, os.path.join(labeled_dir,label,img.split('/')[-1]))
        already_labeled.append(img)
  
    image_paths_sample = random.sample(image_paths,num_images - len(label_types))
    for image in image_paths_sample:
      if image not in already_labeled:
          label = image.split('/')[-1].split('_')[0]
          if label not in label_types:
            label_types[label] = 1
          else:
            label_types[label] += 1
          
          shutil.copy(image, os.path.join(labeled_dir,label,image.split('/')[-1]))
          num_labeled += 1
          already_labeled.append(image)
      if num_labeled==num_images:
        break
    return already_labeled

  # else:
  #   num_labeled = 0
  #   num_images_pos = num_images // 2
  #   all_pos_images = [image for image in image_paths if image.split('/')[-1].split('_')[0] == positive_class]
  #   all_neg_images = [image for image in image_paths if not image.split('/')[-1].split('_')[0] == positive_class]
  #   shuffle(all_pos_images)
  #   pos_images = [all_pos_images[i] for i in range(num_images_pos)]
  #   neg_images = [all_neg_images[i] for i in range(num_images_pos)]
        
  #   for image in pos_images + neg_images:
  #     if image not in already_labeled:
  #       num_labeled += 1
  #       already_labeled.append(image)
  #       if image.split('/')[-1].split('_')[0] == positive_class:
  #         shutil.copy(image, os.path.join(labeled_dir,'positive',image.split('/')[-1]))
  #       else:
  #         shutil.copy(image, os.path.join(labeled_dir,'negative',image.split('/')[-1]))
  #     if num_labeled==num_images:
  #       break
  #   return already_labeled





