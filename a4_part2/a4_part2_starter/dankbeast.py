# Full forward pass, caching intermediate
def full_backward_pass(example, net, activations,  z_hat, z):
    hidden_layer_count = net['hidden_layer_count']
    x = example
    W_1 = net['hidden-#1-W']
    b_1 = net['hidden-#1-b']

    
    gradientDict= {}
    dLdzhat = 2*(z_hat - z)
    print(dLdzhat)
    dLdX = logistic_backprop(dLdzhat, z_hat)
    print("\n\n\nFUCK YO SHIT\n\n\n")
    print(dLdX)
    
    print("\n\nDanker Memes")
    print("dLdX,.shape")
    print(dLdX.shape)
    print("relu(activations[hidden_layer_count]).shape")
    print(relu(activations[hidden_layer_count]).shape)
    print("net['final-W'].shape")
    print(net['final-W'].shape)

    dLdX, dLdW, dLdB = fully_connected_backprop(dLdX, relu(activations[hidden_layer_count]), net['final-W'])
    gradientDict['final-W'] = dLdW
    gradientDict['final-b'] = dLdB
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
    dLdX, dLdW, dLdB = fully_connected_backprop(dLdX, x, W_1)
    gradientDict['hidden-#1-W'] = dLdW
    gradientDict['hidden-#1-b'] = dLdB
 
    return gradientDict
