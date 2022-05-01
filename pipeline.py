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
from torch import optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import importlib
import warnings
from operator import itemgetter
import global_constants as GConst
warnings.filterwarnings("ignore")

from utils import load_config, load_model, load_opt_loss
import tfds
from train_model import train_model_vanilla
from query_strat.query import get_low_conf_unlabeled_batched
from custom_datasets import RESISC_Eval


class Pipeline:

    def __init__(self, config_path) -> None:
        self.config = load_config(config_path)
        self.model_kwargs = self.config['model']
        self.already_labeled = list()
        self.transform = transforms.Compose([
                          transforms.Resize((224,224)),
                          transforms.ToTensor()])
        
    def main(self):
        config = self.config  
        if config['data']['dataset'] == 'tfds':
            dataset_name = config['data']['dataset_name']
            dataset_path = os.path.join(os.getcwd(), dataset_name)
            print("Dataset:", dataset_name)
            positive_class = str(config['data']['positive_class'])
            tfds_prepare = tfds.PrepareData(dataset_name)
            tfds_prepare.download_and_prepare()

            #Initialising data by annotating labeled
            unlabeled_images = list(paths.list_images(GConst.UNLABELED_DIR))
            self.already_labeled = tfds.tfds_annotate(unlabeled_images, 1500, self.already_labeled, labeled_dir=GConst.LABELED_DIR)

            #Train 
            val_dataset = ImageFolder(GConst.VAL_DIR, transform=self.transform)
            test_dataset = ImageFolder(GConst.TEST_DIR, transform = self.transform)


            al_config = config['active_learner']
            al_kwargs = dict(
                            val_dataset = val_dataset, 
                            test_dataset=  test_dataset, 
                            strategy = al_config['strategy'],
                            diversity_sampling = al_config['diversity_sampling'],
                            num_iters = al_config['iterations'],
                            num_labeled = al_config['num_labeled'],
                            limit  = al_config['limit']

                            )
            logs = self.train_al(unlabeled_images, **al_kwargs)


    def train_al(self, unlabeled_images, **al_kwargs):
        iter1 = 0
        val_dataset = al_kwargs['val_dataset']
        test_dataset = al_kwargs['test_dataset']
        num_iters = al_kwargs['num_iters']
        
        logs = {'ckpt_path' : [],
                'graph_logs': []}

        train_config = self.config['train']
        file1 = open(f"/content/drive/MyDrive/{GConst.start_name}_{GConst.diversity_name}.txt","a")
        file1.write(f"{GConst.start_name}__{GConst.diversity_name}\n")
        file1.close()

        while iter1 < num_iters:
            print(f'-------------------{iter1 +1}----------------------')
            iter1 += 1
            model = load_model(**self.model_kwargs)
            optimizer, loss = load_opt_loss(model, self.config)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.925)


            train_kwargs = dict(epochs = train_config['epochs'],
                    opt = optimizer,
                    loss_fn = loss,
                    batch_size = train_config['batch_size'],
                    scheduler = scheduler)
            train_model_vanilla(model, GConst.LABELED_DIR,iter1, val_dataset, test_dataset, **train_kwargs)
            # logs['ckpt_path'].append(ckpt_path)
            # logs['graph_logs'].append(graph_logs)
            low_confs = get_low_conf_unlabeled_batched(model, unlabeled_images, self.already_labeled, train_kwargs, **al_kwargs)
            for image in low_confs:
                if image not in self.already_labeled:
                    self.already_labeled.append(image)
                    label = image.split('/')[-1].split('_')[0]
                    shutil.copy(image, os.path.join(GConst.LABELED_DIR,label,image.split('/')[-1]))

        return logs
