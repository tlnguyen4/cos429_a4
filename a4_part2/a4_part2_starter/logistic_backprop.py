import numpy as np 
from logistic import logistic
def logistic_backprop(dLdy, y):
    # Computes the partial derivative of the loss with respect to the input to
    # the logistic function, given it with respect to the output.
    #
    # dLdy: a scalar double that is the partial derivative of the loss with
    #   respect to y, the output of the logistic function.
    # y: the output of the logistic function (y = logistic(x)).

    # return:
    # [dLdX]: a scalar double that is the partial derivative of the loss with
    # respect to x, the input of the logistic function.
    #
    # Hint: x isn't passed in because it is you can compute this derivative
    # more simply using y. Start by computing the derivative of the logistic
    # function, and then substitute the definition of the logistic function
    # into the derivative.
    #
    # TODO: Implement me!
   
    #we need to compute the derivative dy/dx to get the desired result,
    # this is just the derivative of the sigmoid function apllied to x, which is 
    # logistic(x)(1 - logistic(x))
    dYdX = y@(1-y)
    dLdX = dYdX@dLdy 

    return dLdX
