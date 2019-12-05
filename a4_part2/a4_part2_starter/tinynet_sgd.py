#
# Princeton University, COS 429, Fall 2019
#
# tinynet_sgd.py
#   Trains a tiny (2 hidden-node + 1 output) network with SGD
#
# Inputs:
#   X: datapoints (one per row)
#   z: ground-truth labels (0/1)
#   layers: list of length K. K = the number of hidden layers, excluding the
#   final post-relu layer. The value of each layer is the number of hidden layer neurons.
#   epoch_count: number of epochs to train over
# Output:
#   params: vector of parameters 
import numpy as np
from fully_connected import fully_connected
from logistic import logistic
from relu import relu
# from fully_connected_backprop import fully_connected_backprop
from fully_connected_backprop_gt import fully_connected_backprop
from logistic_backprop import logistic_backprop
from relu_backprop import relu_backprop


def tinynet_sgd(X, z, layers, epoch_count):
    example_count, feature_count = X.shape

    # Randomly initialize the parameters.
    net = initialize_net(layers, feature_count)
    # The value at key i is the intermediate output of the network at
    # layer i. The type is 'any' because the neuron count is variable.
    activations = {}

    # For each epoch, train on all data examples.
    for ep in range(1, epoch_count + 1):
        print('Starting epoch #{} of #{}...\n'.format(ep, epoch_count))
        learning_rate = 0.1 / ep
        # Randomly shuffle the examples so they aren't applied in the
        # same order every epoch.
        permutation = np.random.permutation(example_count)
        X = X[permutation]
        z = z[permutation]

        # Train on each example by making a prediction and doing backprop
        # Note that in a full-featured software package like TensorFLow, 
        # you would pull a batch of images, and all the functions you
        # implemented here would be efficiently batched to reduce repeated
        # work. If your brain is melting thinking through some of the
        # gradients, remember that at least x is a vector, not a matrix. We
        # do care.
        for i in range(example_count):
            # For simplicity set the '0-th' layer activation to be the
            # input to the network. activations[1...hidden_layer_count] are
            # the outputs of the hidden layers of the network before relu.
            # activations[hidden_layer_count+1] is the output of the output
            # layer. The logistic layers and layer don't need to be stored
            # here to compute the derivatives and updates at the end of
            # the network.
            # You will at some point want to access the post relu
            # activations. Just call relu on the activations to get that.
            # Faster networks would cache them, but memory management here
            # is already bad enough that it won't matter much.
            activations[0] = X[i:(i + 1), :]

            # Forward propagation pass to evaluate the network
            # z_hat is the output of the network activations has been
            # updated in-place to contain the network outputs of layer i
            # at activations[i].
            z_hat = full_forward_pass(X[i:(i + 1), :], net, activations)

            print("zhat memes\n")
            print(z_hat)
            print("\n\n\n")

            print("z memes\n")
            print(z[i])
            print("\n\n\n")
            # Backwards pass to evaluate gradients at each layer and update
            # weights. First compute dL/dz_hat here, then use the backprop
            # functions to get the gradients for earlier weights as you
            # move backwards through the network, updating the weights
            # at each step.
            # Note: Study full_forward_pass, defined below, for an example
            # of how the information you need to access can be referenced.
            # [net] contains the current network weights (and is a handle
            # object, so please be careful when editing it). 
            # [activations] contains the responses of the network at each
            # layer activations[0] is the input,
            # activations[1:hidden_layer_count] is the output of each layer
            # before relu has been applied.
            # activations[hidden_layer_count+1] is the final output before
            # it is squished with the logistic function.

            # TODO: Implement me! 
            gradientDict = full_backward_pass(X[i:(i + 1), :], net, activations, z_hat, z[i])
            for parameter in gradientDict:
                net[parameter] = net[parameter] - gradientDict[parameter]*learning_rate
            
    return net


# Full forward pass, caching intermediate
def full_forward_pass(example, net, activations):
    hidden_layer_count = net['hidden_layer_count']
    x = example

    W_1 = net['hidden-#1-W']
    b_1 = net['hidden-#1-b']
    # print("x.shape")
    # print(x.shape)
    # print("W_1.shape")
    # print(W_1.shape)
    # print("b_1.shape")
    # print(b_1.shape)
    activations[1] = fully_connected(x, W_1, b_1) 
    for i in range(2, hidden_layer_count + 1):
        W = net['hidden-#{}-W'.format(i)]
        b = net['hidden-#{}-b'.format(i)]
        # Apply the ith hidden layer and relu and update x.
        activations[i] = fully_connected(relu(activations[i - 1]), W, b)

    W = net['final-W']
    b = net['final-b']
    # Apply the final layer, and then the sigmoid to get zhat.
    # Ignore the unused warning for activations - it is a handle object, so
    # it is pass by reference.
    x = fully_connected(relu(activations[hidden_layer_count]), W, b)
    activations[hidden_layer_count + 1] = x
    z_hat = logistic(x)
    return z_hat



# Full forward pass, caching intermediate
def full_backward_pass(example, net, activations,  z_hat, z):
    hidden_layer_count = net['hidden_layer_count']
    x = example
    W_1 = net['hidden-#1-W']
    b_1 = net['hidden-#1-b']

    
    gradientDict= {}
    dLdzhat = 2*(z_hat - z)
    # print(dLdzhat)
    dLdX = logistic_backprop(dLdzhat, z_hat)
    # print("\n\n\nFUCK YO SHIT\n\n\n")
    # print(dLdX)
    
    # print("\n\nDanker Memes")
    # print("dLdX,.shape")
    # print(dLdX.shape)
    # print("relu(activations[hidden_layer_count]).shape")
    # print(relu(activations[hidden_layer_count]).shape)
    # print("net['final-W'].shape")
    # print(net['final-W'].shape)

    dLdX, dLdW, dLdB = fully_connected_backprop(dLdX, relu(activations[hidden_layer_count]), net['final-W'])
    gradientDict['final-W'] = dLdW
    gradientDict['final-b'] = dLdB
    print(dLdX.shape)
    print(activations[hidden_layer_count].shape)
    dLdX = relu_backprop(dLdX, activations[hidden_layer_count])
    for i in range(hidden_layer_count, 2, -1):
        W = net['hidden-#{}-W'.format(i)]
        b = net['hidden-#{}-b'.format(i)]
        # Apply the ith hidden layer and relu and update x.
        dLdX, dLdW, dLdB = fully_connected_backprop(dLdX, relu(activations[i-1]), W)
        gradientDict['hidden-#{}-W'.format(i)] = dLdW
        gradientDict['hidden-#{}-b'.format(i)] = dLdB
        dLdX = relu_backprop(dLdx, activations[i - 1])

    W_1 = net['hidden-#1-W']
    x = activations[0]
    # x=x[:,0]
    # x = np.transpose(x)
    # dLdX, dLdW, dLdB = fully_connected_backprop(dLdX, x, W_1)
    print("\n\n\n\nSUPREME DANKNESS\n\n")

    dLdX, dLdW, dLdB = fully_connected_backprop(dLdX, x, W_1)
    gradientDict['hidden-#1-W'] = dLdW
    gradientDict['hidden-#1-b'] = dLdB
 
    return gradientDict



def initialize_net(layers, feature_count):
    # Initializes the network with matrices of the correct size containing
    # random weights scaled as in He et al. Also initializes the biases to
    # 0-vectors. Reading this function may be helpful for understanding how
    # the network is structured, but you don't need to edit it.
    hidden_layer_count = len(layers)
    net = {}
    net['layers'] = layers
    net['hidden_layer_count'] = hidden_layer_count

    # Initialize hidden layers
    for i in range(1, hidden_layer_count + 1):
        if i == 1:
            layer_rows = feature_count
        else:
            layer_rows = layers[i - 2]
        layer_cols = layers[i - 1]
        layer_sigma = np.sqrt(2.0 / layer_rows)

        W = np.zeros((layer_rows, layer_cols))
        for col in range(layer_cols):
            W[:, col] = layer_sigma * np.random.normal(0, 1, (layer_rows))
        net['hidden-#{}-W'.format(i)] = W
        net['hidden-#{}-b'.format(i)] = np.zeros((1, layer_cols))
    # Initialize final layer
    layer_size = layers[-1]
    layer_sigma = 1.0 / np.sqrt(layer_size)
    net['final-W'] = layer_sigma * np.random.normal(0, 1, (layer_size, 1))
    net['final-b'] = np.zeros((1, 1))
    return net




