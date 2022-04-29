import torch
import torchvision 
import torch.nn as nn


class ResNet50(nn.Module):
  def __init__(self, **model_kwargs):
    super(ResNet50, self).__init__()
    self.enc= torchvision.models.resnet50(pretrained = True)
    self.enc.fc = nn.Sequential(
    nn.Linear(2048, 10),
    # nn.Sigmoid()
    )

  def forward(self, x):
    x = self.enc(x)
    return x