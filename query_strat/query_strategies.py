import numpy as np
from operator import itemgetter
from scipy.stats import entropy
from query_strat.query_utils import get_samples

'''
Design notes for custom strategies : 

Input : 
confidences:  Confidence values of all the unlabeled images
number : Number of images to be queried

Output :
Paths of all the intelligently queried images
'''

def entropy_based(confidences, number):

    print("Using Entropy Based")

    entropies = entropy(confidences["conf_vals"], axis = 1)

    entropies = (entropies - entropies.mean(axis = 1)) / entropies.std(axis = 1)

    assert len(confidences["loc"]) == len(entropies)

    # path_to_score = dict(zip(confidences["loc"], entropies))

    return entropies

    




def margin_based(confidences, number):

    print("Using Margin Based")

    max_indices = np.argmax(confidences['conf_vals'], axis = 1)
    
    max_vals = confidences[max_indices]

    # making max to a low number that cannot be reselected
    confidences[max_indices] = -1
    second_max_vals = np.max(confidences['conf_vals'], axis = 1)
    difference_array = max_vals - second_max_vals

    difference_array = (difference_array - difference_array.mean(axis = 1)) / difference_array.std(axis = 1)

    assert len(confidences["loc"]) == len(difference_array)

    # path_to_score = dict(zip(confidences["loc"], difference_array))

    return difference_array


def least_confidence(confidences, number):
    
    print("Using Least Confidence")

    difference_array = 1 - np.max(confidences['conf_vals'], axis = 1)

    difference_array = (difference_array - difference_array.mean(axis = 1)) / difference_array.std(axis = 1)
    
    assert len(confidences["loc"]) == len(difference_array)

    # path_to_score = dict(zip(confidences["loc"], difference_array))

    return difference_array
