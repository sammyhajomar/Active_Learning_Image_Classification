from ast import Index
from copy import copy
from distutils.command.config import config
import gc
import sys
import os 
import time
import imutils
from random import shuffle 
import pandas as pd
from imutils import paths 
import matplotlib.pyplot as plt
from PIL import Image
import pathlib
from pathlib import Path
import argparse
import shutil
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import importlib
import warnings
from operator import itemgetter
import global_constants as GConst
warnings.filterwarnings("ignore")

from utils import (copy_data, load_config, load_model, load_opt_loss, initialise_data_dir, annotate_data, get_num_files)
import tfds
from train_model import train_model_vanilla
from query_strat.query import get_low_conf_unlabeled_batched
from custom_datasets import RESISC_Eval

def adhoc_copy(unlabeled_paths):
    imgs = unlabeled_paths['image_paths'].values[:4]
    for i in range(len(imgs)):
        if i %2 == 0:
            shutil.copy(imgs[i], os.path.join(GConst.LABELED_DIR, 'negative'))
        else:
            shutil.copy(imgs[i], os.path.join(GConst.EVAL_DIR, 'negative'))
    print('ADHOC DONE : ', len(imgs))

class Pipeline:

    def __init__(self, config_path) -> None:
        self.config = load_config(config_path)
        # initialise_data_dir()

        model_kwargs = self.config['model']
        self.model = load_model(**model_kwargs)
        self.optim, self.loss = load_opt_loss(self.model, self.config)
        self.already_labeled = list()
        self.transform = transforms.Compose([
                          transforms.Resize((224,224)),
                          transforms.ToTensor(),
                          transforms.Normalize((0, 0, 0),(1, 1, 1))])
        
        # self.labeler = Labeler(self.config)

    def main(self):
        config = self.config
        if config['data']['dataset'] == 'resisc':
            positive_class = config['data']['positive_class']
            resisc.download_and_prepare()
            #Initialising data by annotating labeled and eval
            unlabeled_images = list(paths.list_images(GConst.UNLABELED_DIR))
            print(len(unlabeled_images))
            self.already_labeled = resisc.resisc_annotate(unlabeled_images, 100, self.already_labeled, positive_class, labeled_dir=GConst.EVAL_DIR, val=True) 
            self.already_labeled = resisc.resisc_annotate(unlabeled_images, 50, self.already_labeled, positive_class, labeled_dir=GConst.LABELED_DIR)
            print("Total Eval Data: Positive {} Negative {}".format(get_num_files("eval_pos"),get_num_files('eval_neg')))
            print("Total Labeled Data: Positive {} Negative {}".format(get_num_files("positive"),get_num_files('negative')))

            #Train 
            eval_dataset = RESISC_Eval(GConst.UNLABELED_DIR, positive_class)
            val_dataset = ImageFolder(GConst.EVAL_DIR, transform = self.transform)

            train_config = config['train']
            train_kwargs = dict(epochs = train_config['epochs'],
                                opt = self.optim,
                                loss_fn = self.loss, 
                                batch_size = train_config['batch_size'],
                                )

            al_config = config['active_learner']
            al_kwargs = dict(
                            eval_dataset = eval_dataset, 
                            val_dataset=  val_dataset, 
                            strategy = al_config['strategy'],
                            positive_class = positive_class,
                            num_iters = al_config['iterations'],
                            num_labeled = al_config['num_labeled'],
                            limit  = al_config['limit']
                            )
            logs = self.train_al(self.model, unlabeled_images, train_kwargs, **al_kwargs)
        
        elif config['data']['dataset'] == 'tfds':
            dataset_name = config['data']['dataset_name']
            dataset_path = os.path.join(os.getcwd(), dataset_name)
            print("Dataset:", dataset_name)
            positive_class = str(config['data']['positive_class'])
            tfds_prepare = tfds.PrepareData(dataset_name,positive_class)
            tfds_prepare.download_and_prepare()

            #Initialising data by annotating labeled and eval
            unlabeled_images = list(paths.list_images(GConst.UNLABELED_DIR))
            self.already_labeled = tfds.tfds_annotate(unlabeled_images, 100, self.already_labeled, positive_class, labeled_dir=GConst.EVAL_DIR, val=True) 
            self.already_labeled = tfds.tfds_annotate(unlabeled_images, 50, self.already_labeled, positive_class, labeled_dir=GConst.LABELED_DIR)
            print("Total Eval Data: Positive {} Negative {}".format(get_num_files("eval_pos"),get_num_files('eval_neg')))
            print("Total Labeled Data: Positive {} Negative {}".format(get_num_files("positive"),get_num_files('negative')))

            #Train 
            if os.path.exists(GConst.VALID_DIR):
              eval_dataset = RESISC_Eval(GConst.VALID_DIR, positive_class)
            else:
              eval_dataset = RESISC_Eval(GConst.UNLABELED_DIR, positive_class)
            val_dataset = ImageFolder(GConst.EVAL_DIR, transform = self.transform)

            train_config = config['train']
            train_kwargs = dict(epochs = train_config['epochs'],
                                opt = self.optim,
                                loss_fn = self.loss, 
                                batch_size = train_config['batch_size'],
                                )

            al_config = config['active_learner']
            al_kwargs = dict(
                            eval_dataset = eval_dataset, 
                            val_dataset=  val_dataset, 
                            strategy = al_config['strategy'],
                            positive_class = positive_class,
                            num_iters = al_config['iterations'],
                            num_labeled = al_config['num_labeled'],
                            limit  = al_config['limit']
                            )
            logs = self.train_al(self.model, unlabeled_images, train_kwargs, **al_kwargs)


        
        elif config['data']['dataset'] == 'csv':

            self.df = pd.read_csv(config['data']['path'])
            df = self.df.copy()
            query_image = df[df['status'] == 'query'][GConst.IMAGE_PATH_COL].values
            unlabeled_paths = df[df['status'] != 'query']
            unlabeled_paths_lis = unlabeled_paths[GConst.IMAGE_PATH_COL].values
            num_labeled = config['active_learner']['num_labeled']
            self.preindex = self.config['active_learner']['preindex']
            if self.preindex:
                self.index = Indexer(unlabeled_paths_lis, self.model, img_size=224, 
                                     index_path = None)
            
            if len(query_image) > 1:
                split_ratio = int(0.9 * len(query_image)) #TODO make this an arg
                annotate_data(query_image[split_ratio:], 'eval_pos')
                annotate_data(query_image[:split_ratio], 'positive')
            else:
                annotate_data(query_image, 'positive')
                annotate_data(query_image, 'eval_pos')
            
            if self.preindex: 
                #FAISS Fetch
                similar_imgs = self.index.process_image(query_image[0], n_neighbors=num_labeled *2) #hardcoding sending only the first image here from query images
                train_init = similar_imgs[:num_labeled]
                val_init = similar_imgs[num_labeled:]
                self.labeler.label(train_init, is_eval = False)
                self.labeler.label(val_init, is_eval = True)
                # self.sl.label(train_init, is_eval=False)
                # self.sl.label(val_init, is_eval = True)
                self.already_labeled.extend(similar_imgs)
            else:
                random_init_imgs = unlabeled_paths.sample(num_labeled * 2)[GConst.IMAGE_PATH_COL].values
                train_init = random_init_imgs[:num_labeled]
                val_init = random_init_imgs[num_labeled:]

                # self.sl.label(train_init, is_eval=False)
                # self.sl.label(val_init, is_eval = True)

                self.labeler.label(train_init, is_eval = False)
                self.labeler.label(val_init, is_eval = True)
                self.already_labeled.extend(random_init_imgs)


            #swipe_labeler -> label random set of data -> labeled pos/neg. Returns paths labeled
            print("Total annotated valset : {} Positive {} Negative".format(get_num_files("eval_pos"),get_num_files('eval_neg')))
            print("Total Labeled Data: Positive {} Negative {}".format(get_num_files("positive"),get_num_files('negative')))
            
            train_config = config['train']    
            #data is ready ,start training and AL   
            train_kwargs = dict(epochs = train_config['epochs'],
                                opt = self.optim,
                                loss_fn = self.loss, 
                                batch_size = train_config['batch_size']
                                )
                                
            al_config = config['active_learner']
            al_kwargs = dict(
                            strategy = al_config['strategy'],
                            num_iters = al_config['iterations'],
                            num_labeled = al_config['num_labeled'],
                            limit  = al_config['limit']
                            )

            adhoc_copy(unlabeled_paths)

            logs = self.train_al_csv(self.model, unlabeled_paths_lis, train_kwargs, **al_kwargs)


    def train_al_csv(self, model, unlabeled_images, train_kwargs, **al_kwargs):
        iter = 0
        num_iters = al_kwargs['num_iters']

        logs = {'ckpt_path' : [],
                'graph_logs' : []}
        
        while iter < num_iters:
            print(f'-------------------{iter +1}----------------------')
            iter+=1
            ckpt_path, graph_logs = train_model_vanilla(self.model, GConst.LABELED_DIR, **train_kwargs)
            logs['ckpt_path'].append(ckpt_path)
            logs['graph_logs'].append(graph_logs)
            low_confs = get_low_conf_unlabeled_batched(model, unlabeled_images, self.already_labeled, **al_kwargs)
            # self.sl.label(low_confs, is_eval = False)
            self.labeler.label(low_confs, is_eval = False)

            self.already_labeled.extend(low_confs)
            print("Total Labeled Data: Positive {} Negative {}".format(get_num_files('positive'), get_num_files('negative')))

        return logs



    def train_al(self, model, unlabeled_images, train_kwargs, **al_kwargs):
        iter1 = 0
        eval_dataset = al_kwargs['eval_dataset']
        val_dataset = al_kwargs['val_dataset']
        num_iters = al_kwargs['num_iters']
        positive_class = al_kwargs['positive_class']
        
        logs = {'ckpt_path' : [],
                'graph_logs': []}

        while iter1 < num_iters:
            print(f'-------------------{iter1 +1}----------------------')
            iter1 += 1
            ckpt_path, graph_logs = train_model_vanilla(self.model, GConst.LABELED_DIR, eval_dataset, val_dataset, **train_kwargs)
            logs['ckpt_path'].append(ckpt_path)
            logs['graph_logs'].append(graph_logs)
            low_confs = get_low_conf_unlabeled_batched(model, unlabeled_images, self.already_labeled, train_kwargs, **al_kwargs)
            for image in low_confs:
                if image not in self.already_labeled:
                    self.already_labeled.append(image)
                if image.split('/')[-1].split('_')[0] == positive_class:
                    shutil.copy(image, os.path.join(GConst.LABELED_DIR,'positive',image.split('/')[-1]))
                else:
                    shutil.copy(image, os.path.join(GConst.LABELED_DIR,'negative',image.split('/')[-1]))
            print("Total Labeled Data: Positive {} Negative {}".format(get_num_files('positive'), get_num_files('negative')))

        return logs





