import numpy as np


def logistic(x):
    """The logistic "sigmoid" function
    """
    return 1/(1 + np.exp(-x)) 
