import os 
import numpy as np
from torchvision import transforms
from operator import itemgetter
from tqdm import tqdm
import torch
from query_strat.diversity_sampling import pick_top_n

from custom_datasets import AL_Dataset
import query_strat.query_strategies as query_strategies

def get_low_conf_unlabeled_batched(model, image_paths, already_labeled, train_kwargs, **al_kwargs):

  strategy = al_kwargs['strategy']
  num_labeled = al_kwargs['num_labeled']
  limit = al_kwargs['limit']

  loss_fn = train_kwargs['loss_fn']

  confidences =  []
  unlabeled_imgs = [os.path.expanduser(img) for img in image_paths if img not in already_labeled]
  t = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0, 0, 0),(1, 1, 1))])
 
  dataset = AL_Dataset(unlabeled_imgs, limit, t)
  unlabeled_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=4, batch_size=64) #add num workers arg

  confidences = {'conf_vals': [],
                 'loc' : []}

  batch_bar = tqdm(total=len(unlabeled_loader), dynamic_ncols=True, leave=False, position=0, desc='Get Most Uncertain Samples') 
  model.eval()
  with torch.no_grad():
    for _, data in enumerate(unlabeled_loader):
      image, loc = data
      outputs = model(image.to('cuda'))

      # outputs = torch.argmax(outputs,axis=1)
      outputs = outputs.detach().cpu().numpy() #.tolist()

      confidences['loc'].extend(loc)
      
      confidences['conf_vals'].append(outputs)

      batch_bar.update()

  batch_bar.close()

  confidences['conf_vals'] = np.concatenate(confidences['conf_vals'])
  confidences['loc'] = np.array(confidences['loc'])


  uncertainty_scores = getattr(query_strategies, strategy)(confidences, num_labeled)
  # close to 1 is more uncertain
  
  # now take uncertainties and use it to perform diversity sampling.
  selected_filepaths = pick_top_n(uncertainty_scores, confidences['loc'], num_labeled)

  raise NotImplementedError

