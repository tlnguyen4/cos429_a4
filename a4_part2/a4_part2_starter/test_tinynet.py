#
# Princeton University, COS 429, Fall 2019
#
# test_tinynet.py
#   Test tinynet on some simple data in 'training_pacman.txt'
#   and 'test_pacman.txt'
#
import numpy as np
import matplotlib.pyplot as plt
from tinynet_sgd import tinynet_sgd
from tinynet_predict import tinynet_predict


def test_tinynet():
    run_test([6,6,6], 1, 20)
    run_test([6,6,6], 5, 20)
    run_test([6,6,6], 10, 20)
    run_test([6,6,6], 20, 20)
    run_test([6,6,6], 100, 20)
def run_test(layers, epoch_count, numIterations):
    # Load the training data
    # numIterations = 20
    sum_testing = 0
    sum_training = 0

    testing_accuracies = []
    training_accuracies = []

    min_training = 100
    min_testing = 100
    max_training = 0
    max_testing= 0
    #params
    # layers = [9,6,3]
    # epoch_count = 6


    for i in range(0, numIterations):
        training = np.loadtxt('training_pacman.txt')
        
        # Do the training
        example_count = training.shape[0]
        X = training[:, :2]
        z = training[:, 2]

        net = tinynet_sgd(X, z, layers, epoch_count)

        # Apply the learned model to the training data and print out statistics
        predicted = tinynet_predict(X, net)
        training_accuracy = np.sum(predicted == z) / example_count
        training_accuracies.append(training_accuracy)
        min_training = min(min_training, training_accuracy)
        max_training = max(max_training, training_accuracy)
        sum_training += training_accuracy
        # print('training_accuracy =', training_accuracy)

        # Plot "ground truth" and predictions
        # plot_classes(training, z, 'Training ground truth', 1)
        # plot_classes(training, predicted, 'Training predicted', 2)

        # Apply the learned model to the test data
        testing = np.loadtxt('test_pacman.txt')
        example_count = testing.shape[0]
        X = testing[:, :2]
        z = testing[:, 2]
        predicted = tinynet_predict(X, net)
        testing_accuracy = np.sum(predicted == z) / example_count
        testing_accuracies.append(testing_accuracy)
        min_testing = min(min_testing, testing_accuracy)
        max_testing = max(max_testing, testing_accuracy)
        sum_testing += testing_accuracy
        # print('testing_accuracy = ', testing_accuracy)
        # print("\n")
        # Plot "ground truth" and predictions
        # plot_classes(testing, z, 'Testing ground truth', 3)
        # plot_classes(testing, predicted, 'Testing predicted', 4)
    ave_test = sum_testing/numIterations
    ave_train = sum_training/numIterations
    print("\\newline\n\\newline\n\\textbf{Test Parameters: Num Layers: "+str(len(layers))+",   Layers: "+str(layers)+",   Epochs: "+str(epoch_count)+",   Num Trials: "+str(numIterations))
    print("\\newline\nResults: ave test: "+str(ave_test)+", ave train: "+str(ave_train)+",  range test: ("+str(min_testing)+","+str(max_testing)+"), range train: ("
        +str(min_training)+","+str(max_training)+")}")
    # print("\\newline \n\\newline \n training accuracies list:"+str(training_accuracies))
    # print("\\newline \n testing accuracies list:"+str(testing_accuracies))
def plot_classes(data, z, name, num):
    """Create a scatterplot of the given data.  It is assumed that the input
       data is 2-dimensional.

    Args:
        data: datapoints (one per row, only first two columns used)
        z: labels (0/1)
        name: name of the figure
        num: number of the figure
    """
    plt.figure(num)
    plt.title(name)
    positive = data[z > 0]
    negative = data[z == 0]
    plt.scatter(positive[:, 0], positive[:, 1], c='red', alpha=0.5)
    plt.scatter(negative[:, 0], negative[:, 1], c='blue', alpha=0.5)
    plt.show()


if __name__ == '__main__':
    test_tinynet()
