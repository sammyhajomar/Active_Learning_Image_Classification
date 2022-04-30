import torch
import torchvision 
import torch.nn as nn


class ResNet50(nn.Module):
  def __init__(self, **model_kwargs):
    super(ResNet50, self).__init__()
    self.enc= torchvision.models.resnet50(pretrained = True)
    self.enc.fc = nn.Identity() 
    self.true_fc = nn.Linear(2048, 10)
        

  def forward(self, x, want_embeddings = False):

    embedding = self.enc(x)
    
    out = self.true_fc(embedding)

    if want_embeddings == True:
      return out, embedding
    else:
      return out

