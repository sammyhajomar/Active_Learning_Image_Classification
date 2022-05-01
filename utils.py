from json import encoder
import yaml
import torch
import importlib
from torch import optim
import torch.nn as nn
import os 
import shutil
from tqdm import tqdm
import global_constants as GConst
from imutils import paths
import sys


def load_config(config_path):
    """" Loads the config file into a dictionary. """
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

def load_model(**model_kwargs):
    device = model_kwargs['device']

    if device != "cuda":
        raise NotImplementedError("Currently supporting only cuda backends, please change device in the config to cuda.")

    model = model_kwargs['model']
    model = getattr(importlib.import_module('model'), model)(**model_kwargs)
    model.to(device)

    return model


def load_opt_loss(model, config, is_ssl = False):
    """Fetches optimiser and loss fn params from config and loads"""
    opt_params = config['train']['optimizer']
    loss_params = config['train']['loss_fn']
    ssl_config = config['model'].get('ssl', {})
    loss_kwargs = {k:loss_params[k] for k in loss_params if k!='name'}
    if ssl_config:
        encoder_lr = ssl_config['encoder']['e_lr'] if ssl_config['encoder']['train_encoder'] else 0
        optimizer = getattr(optim, opt_params['name'])(
            [
                {"params": model.encoder.parameters(), "lr": encoder_lr},
                {"params": model.linear_model.parameters(), "lr": ssl_config['classifier']['c_lr']},
            ],
            **opt_params.get('config', {})
        )
    else:
        optimizer = getattr(optim, opt_params['name'])(
                    model.parameters(), momentum=0.9, weight_decay=1e-4, **opt_params.get('config', {}))

    loss_fn = getattr(nn, loss_params['name'])(**loss_kwargs)

    return optimizer, loss_fn

def copy_data(paths, folder):
    for image in tqdm(paths):
        shutil.copy(image, folder)
    print('Data Copied to {}'.format(folder))