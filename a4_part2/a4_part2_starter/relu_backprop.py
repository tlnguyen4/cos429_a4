import numpy as np
def relu_backprop(dLdy, x):
    # Backpropogates the partial derivatives of loss with respect to the output
    # of the relu function to compute the partial derivatives of the loss with
    # respect to the input of the relu function. Note that even though relu is
    # applied elementwise to matrices, relu_backprop is consistent with
    # standard matrix calculus notation by making the inputs and outputs row
    # vectors.
    # dLdy: a row vector of doubles with shape [1, N]. Contains the partial
    #   derivative of the loss with respect to each element of y = relu(x).
    # x: a row vector of doubles with shape [1, N]. Contains the elements of x
    #   that were passed into relu().
    # return:
    # [dLdX]: a row vector of doubles with shape [1, N]. Should contain the
    #   partial derivative of the loss with respect to each element of x.
    # TODO: Implement me!

    #derivate of relu is 1 for x > 0, 0 otherwise. Undefined for x = 0 
    dYdx = (x > 0)
    #chain rule using hadamard product to get entry-wise partial derivative
    dLdX = np.multiply(dYdx, dLdy)

    return dLdX
