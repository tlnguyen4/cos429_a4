import numpy as np

def relu(x):
    # x: a 2-D double array with arbitrary shape.
    # return: relu(x) as described in class applied elementwise.
    # TODO: Implement me!
    return np.maximum(x, 0)
