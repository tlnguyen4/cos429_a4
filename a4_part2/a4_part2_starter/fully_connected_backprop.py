import numpy as np
def fully_connected_backprop(dLdU, x, W):
    # Computes the derivates of the loss with respect to the input x,
    # weights W, and biases b.
    #
    # dLdU: a row-vector with shape [1, neuron_count] that contains the
    #   derivative of the loss with respect to each element of the output of
    #   the fully connected layer.
    # x: a row-vector with shape [1, feature_count]. The input example that was
    #   passed into the layer. This is either the original training feature
    #   vector, if this is the first layer of the network, or the activations
    #   of the previous layer of the network if this layer is further in.
    # W: a matrix with shape [feature_count, neuron_count]. The current weight
    #   matrix at this layer.
    # [not provided: b]. A row vector with shape [1, neuron_count]. The current
    #   bias vector at this layer. Hint: We don't provide it because the
    #   derivatives work out such that it isn't necessary.
    # return:
    # [dLdX]: a row-vector with shape [1, feature_count] (the same shape as
    #   x). Each element of this vector should contain the partial derivative
    #   of L with respect to the corresponding value of the input x.
    # [dLdW]: a matrix with shape [feature_count, neuron_count] (the same shape
    #   as W). Each element should contain the partial derivative of L with
    #   respect to the corresponding value of W. Hint: While a matrix shape on
    #   output is necessary to update W, much of the matrix calculus can be
    #   simplified if you work with a row vector of shape
    #   [1,feature_count*neuron_count] internally.
    # [dLdB]: a row-vector with shape [1, neuron_count] (the same shape as b).
    #   Each element should contain the partial derivative of L with respect to
    #   the corresponding value of b.
    # TODO: Implement me!

    #U = Wx + b
    #U is the output of the fully connected layer thus, the output of the weights applied to x
    # we take the derivative with respect to x
    dUdx = W

    #get derivative of loss with respect to x
    dLdX = np.transpose(dUdx@np.transpose(dLdU))

    #derivative with respect to x. Have to transpose for hadamard product with dLdU
    dUdw = np.transpose(x)

    #take hadamard product to get dLdw
    dLdW =  np.multiply(dUdw, dLdU)    
   
    # each b entry is just a constant, so we just need a matrix of 1's of the correct size
    #technically not even necessary to directly compute dUdb, but we do it for sake of clarity 
    dUdb = np.ones((1,W.shape[1]))  

    #take hadamard product
    dLdB = np.multiply(dUdb,dLdU)

    return dLdX, dLdW, dLdB
