#
# Princeton University, COS 429, Fall 2019
#
# tinynet_predict.py
#   Given a tinynet model and some new data, predicts classification
#
# Inputs:
#   X: datapoints (one per row)
#   params: vector of parameters 
# Output:
#   z: predicted labels (0/1)
#
import numpy as np
from tinynet_sgd import full_forward_pass


def tinynet_predict(X, net):
    # hidden_layer_count = net['hidden_layer_count']
    example_count, _ = X.shape
    z_hat = np.zeros(example_count)
    # error memes here
   
    for ei in range(example_count):
        x = X[ei, :]
        # Set z_hat[ei] by propogating x through the network.
        # This task is a warm-up: a superset of the functionality required
        # here is already implemented in the full_forward_pass() function
        # in tinynet_sgd.py
        # TODO: Implement me!
        activations = {}
        activations[0] = x 
        z_hat[ei] = full_forward_pass(x, net, activations)
    z = (z_hat > 0.5)
    return z
