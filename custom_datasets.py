from torch.utils.data import Dataset
from PIL import Image
from random import shuffle
from torchvision import transforms

class AL_Dataset(Dataset):

  def __init__(self, unlabeled_imgs,  limit, transform = None):
    self.unlabeled_imgs = unlabeled_imgs
    self.transform =  transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor()])
    if limit == -1:
      print("Getting confidences for entire unlabeled dataset")
    else:
      print(f"Getting Confidences from random {limit} data")
      shuffle(self.unlabeled_imgs)
      self.unlabeled_imgs = self.unlabeled_imgs[:limit]
    self.transform = transform

  def __len__(self):
    return len(self.unlabeled_imgs)

  def __getitem__(self, index):
    img_path = self.unlabeled_imgs[index]
    img = Image.open(img_path).convert('RGB')
    
    if self.transform:
      img = self.transform(img)
    return img, img_path
