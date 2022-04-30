import numpy as np

def pick_top_n(uncertainty_scores, filepaths, number):
    #this will pick the top n samples regardless of diversity

    top_indices = uncertainty_scores.argsort()[::-1][:number]

    return list(filepaths[top_indices])


def iterative_proximity_sampling(uncertainty_scores, filepaths, number):
    raise NotImplementedError


def clustering_sampling(uncertainty_scores, filepaths, number):
  raise NotImplementedError


def random_sampling(uncertainty_scores, filepaths, number):
  raise NotImplementedError
