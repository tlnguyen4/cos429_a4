"""
 Princeton University, COS 429, Fall 2019
"""
import numpy as np

def s(z):
    return 1 / (1 + np.power(np.e, -z))

def z_hat(xi, w):
    return s(xi @ w)

def logistic_sgd(X, z, num_epochs):
    """Performs L2-regularized logistic regression via Stochastic Gradient Descent 

    Args:
        X: datapoints (one per row, should include a column of ones
                       if the model is to have a constant)
        z: labels (0/1)
        num_epochs: number of epochs to train over

    Returns:
        params: vector of parameters 
    """
    num_pts, num_vars = X.shape
    print("num_pts: ", num_pts)
    print("num_vars: ", num_vars)

    # Initial (random) estimate of params.
    mean = 0
    sigma = 1 / np.sqrt(num_vars / 2)
    params = np.random.normal(mean, sigma, num_vars)

    # Loop over epochs
    for ep in range(1, num_epochs+1):
        # Permute the data rows
        permutation = np.random.permutation(num_pts)
        X = X[permutation]
        z = z[permutation]
        # Iterate over the points
        for i in range(num_pts):
            # Fill in here
            z_hat_i = z_hat(X[i], params)
            gradient = 2 * (z_hat_i - z[i]) * z_hat_i * (1 - z_hat_i) * X[i]
            params = params - (1 / num_epochs) * gradient
            pass
    return params


