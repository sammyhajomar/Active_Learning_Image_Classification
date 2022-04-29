import torch
import torchvision 
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from random import shuffle
import datetime 
import re
from tqdm import tqdm
import sklearn.metrics as metrics
import numpy as np
import datetime 
import os
import shutil
import re
import time
from imutils import paths
import global_constants as GConst


def calculate_metrics(preds, trues, cm=False):

    print("preds",preds)
    print("trues",trues)
    fscore = metrics.f1_score(trues, preds)
    print("fscore",fscore)
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]  
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
    trues = [int(true.item()) for true in trues]
    ### Summing over all correct predictions
    acc = np.sum(acc) / len(preds)
    
    if cm:
      print("Confusion Matrix:")
      print(metrics.confusion_matrix(
            trues,preds,labels=[0,1]
        ))
    fscore = metrics.f1_score(trues, preds,average='binary')
    return (acc * 100), fscore
    

def evaluate_al(model, eval_dataset, loss_fn):
  
  testloader = torch.utils.data.DataLoader(eval_dataset, batch_size=64,
                                      shuffle=True, num_workers=4)

  epoch_loss = []
  preds = []
  trues = []
  batch_bar = tqdm(total=len(testloader), dynamic_ncols=True, leave=False, position=0, desc='Validation') 
  for data in testloader:
      images, labels = data
      labels = labels.to(torch.float32)
      labels = labels.reshape((labels.shape[0], 1)).to('cuda')
      outputs = model(images.to('cuda'))
      
      loss = loss_fn(outputs, labels)
      loss = loss.item()
      epoch_loss.append(loss)

      preds.extend(outputs.detach().cpu().numpy())
      trues.extend(labels.detach().cpu().numpy())
      batch_bar.update()
  batch_bar.close()
  acc, fscore = calculate_metrics(preds, trues, True)
  print(f"On the validation set: \n Valid Accuracy: {acc} \n F1 Score: {fscore}")

def val_model_vanilla(model,val_dataset, val_loader, loss_fn,batch_size):
  
  epoch_loss = []
  epoch_acc = []
  epoch_f1 = []
  num_correct_val = 0
  total_loss_val = 0
  model.eval()
  batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc='Val') 
  for i,(images,labels) in enumerate(val_loader):
      images = images.to('cuda')
      labels = labels.to('cuda')

      with torch.no_grad():
          outputs = model(images)
      
      loss = loss_fn(outputs, labels)
      loss = loss.item()
      epoch_loss.append(loss)

      outputs = torch.argmax(outputs, axis=1)
      num_correct_val += int((outputs == labels).sum())
      total_loss_val += float(loss)
      
      batch_bar.set_postfix(
          acc="{:.04f}%".format(100 * num_correct_val / ((i + 1) * batch_size)),
          loss="{:.04f}".format(float(total_loss_val / (i + 1))))

      batch_bar.update()
  batch_bar.close()

  valid_acc = num_correct_val / len(val_dataset)
  print("Val Acc: {:.04f}%, Train Loss {:.04f},".format(100 * valid_acc,float(total_loss_val / len(val_dataset))))

  return num_correct_val,total_loss_val


def train_model_vanilla(model, train_datapath, val_dataset=None, test_dataset=None, **train_kwargs):

  num_epochs = train_kwargs['epochs']
  batch_size = train_kwargs['batch_size']
  optimizer = train_kwargs['opt']
  loss_fn = train_kwargs['loss_fn']

  t = transforms.Compose([
                          transforms.Resize((224,224)),
                          transforms.RandomHorizontalFlip(p=0.5),
                          transforms.RandomRotation(15),
                          transforms.RandomCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize((0, 0, 0),(1, 1, 1))
  ])

  val_t = transforms.Compose([
                          transforms.Resize((224,224)),
                          transforms.ToTensor(),
                          transforms.Normalize((0, 0, 0),(1, 1, 1))])


  train_dataset = ImageFolder(train_datapath, transform=t)
  val_dataset = ImageFolder(GConst.VAL_DIR, transform = val_t)
  train_imgs = train_dataset.imgs

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=4)

  print("Training")
  # print('{:<10s}{:>4s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}'.format("Epoch", "Train Loss", "Train Acc", "Train F1", "Val Loss", "Val Acc", "Val F1"))

  graph_logs = {}
  graph_logs['val_f1'] = []
  graph_logs['len_data'] = []
  for epoch in range(num_epochs):
    epoch_loss = []
    epoch_acc = []
    epoch_f1 = []
    num_correct = 0
    total_loss = 0
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        epoch_loss.append(loss.item())

        outputs = torch.argmax(outputs, axis=1)
        num_correct += int((outputs == labels).sum())
        total_loss += float(loss)

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))


        # acc, fscore = calculate_metrics(outputs, labels)
        # epoch_acc.append(acc)
        # epoch_f1.append(fscore)

        loss.backward()
        optimizer.step()
        batch_bar.update()
    batch_bar.close()


    print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
        epoch + 1,
        num_epochs,
        100 * num_correct / (len(train_loader) * batch_size),
        float(total_loss / len(train_loader)),
        float(optimizer.param_groups[0]['lr'])))

    # val_fscore, val_acc, val_loss = val_model_vanilla(model, val_loader, loss_fn)
    num_correct_val,total_loss_val = val_model_vanilla(model,val_dataset, val_loader, loss_fn,batch_size)
    


    # avg_acc = np.mean(epoch_acc) 
    # avg_loss = np.mean(epoch_loss)
    # avg_fscore = np.mean(epoch_f1)
    
    # print('{:<10d}{:>4.2f}{:>13.2f}{:>13.2f}{:>13.2f}{:>13.2f}{:>13.2f}'.format(epoch, avg_loss, avg_acc, avg_fscore, val_loss, val_acc, val_fscore))

    # print("Small Eval F1 Score: ",val_fscore,"Total Data Used :",len(list(paths.list_images(GConst.LABELED_DIR))))

  # graph_logs['val_f1'].append(val_fscore)
  # graph_logs['len_data'].append(len(list(paths.list_images(GConst.LABELED_DIR))))
  
  # fscore = 0.0
  # auc = 0.0

  # timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
  # training_size = str(len(train_imgs))
  # accuracies = str(fscore)+"_"+str(auc)
  # model_path = "checkpoints/"+timestamp+ accuracies+ "_" + training_size+".params"
  # torch.save(model.state_dict(), model_path)
  
  print("Since our model has become confident enough, testing on validation data")
  # if val_dataset:
  #   evaluate_al(model, val_dataset, loss_fn)
  # return model_path, graph_logs

