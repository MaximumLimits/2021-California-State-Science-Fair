from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np

from science_fair_8th_9th.images import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        h1 = np.dot(X, W1) + b1
        X2 = np.maximum(0, h1)
        scores = np.matmul(X2, W2) + b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        scores -= np.max(scores, axis=1, keepdims=True)  # avoid numeric instability
        exps = np.exp(scores)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        correct_probs = probs[range(N), y]
        loss = np.sum(-np.log(correct_probs))
        loss /= N
        loss += (reg * np.sum(W1**2)) + (reg * np.sum(W2**2))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dS = probs
        dS[range(N), y] -= 1

        grads['W2'] = (np.dot(X2.T, dS) / N) + reg * W2 * 2
        grads['b2'] = np.sum(dS, axis=0) / N

        dX2 = dS.dot(W2.T)
        dh1 = (h1 > 0) * dX2
        grads['W1'] = np.matmul(X.T, dh1) / N + reg * W1 * 2
        grads['b1'] = np.sum(dh1, axis=0) / N

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            random_indices = np.random.choice(range(num_train), batch_size)
            X_batch = X[random_indices, :]
            y_batch = y[random_indices]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            for param in self.params:
                self.params[param] -= learning_rate * grads[param]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # W1, W2 = self.params["W1"], self.params["W2"]
        # b1, b2 = self.params["b1"], self.params["b2"]
        # fc1 = np.matmul(X, W1) + b1
        # X2 = np.maximum(0, fc1)
        # scores = np.matmul(X2, W2) + b2
        scores = self.loss(X)
        y_pred = scores.argmax(axis=1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred


X_train, y_train = get_images('train')
X_test, y_test = get_images('test')
X_val, y_val = get_images('val')
print('Done fetching data.')
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train = (X_train - X_mean) / X_std
# X_test = (X_test - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
print('Done preprocessing data.')
print()

# LR: 8.496424e-02 | Reg: 2.039568e-04 | Train: 0.996606 | Val: 0.618902
input_size = X_train[0].size
while False:
    lr = 10 ** np.random.uniform(-2.5, -1)
    reg = 10 ** np.random.uniform(-4, -3)  # -4 seems to work the best

    network = TwoLayerNet(input_size, 100, 6)
    stats = network.train(X_train, y_train, X_val, y_val,
                          learning_rate=lr, learning_rate_decay=0.95,
                          reg=reg, num_iters=800, batch_size=200,
                          verbose=False)

    y_train_pred = network.predict(X_train)
    y_val_pred = network.predict(X_val)

    train_accuracy = np.mean(y_train_pred == y_train)
    val_accuracy = np.mean(y_val_pred == y_val)

    print('LR: %e | Reg: %e | Train: %f | Val: %f' % (lr, reg, train_accuracy, val_accuracy))

# Without data preprocessing
# best_net = TwoLayerNet(input_size, 500, 6)
# best_net.train(X_train, y_train, X_val, y_val,
#               learning_rate=2.00000e-04, reg=4.00000e-02,
#                num_iters=1500, verbose=True)
#
# iteration 0 / 1500: loss 1.792462
# iteration 100 / 1500: loss 1.670217
# iteration 200 / 1500: loss 1.612969
# iteration 300 / 1500: loss 1.528777
# iteration 400 / 1500: loss 1.356371
# iteration 500 / 1500: loss 1.255917
# iteration 600 / 1500: loss 1.414482
# iteration 700 / 1500: loss 1.280708
# iteration 800 / 1500: loss 1.319135
# iteration 900 / 1500: loss 1.251605
# iteration 1000 / 1500: loss 1.358585
# iteration 1100 / 1500: loss 1.040733
# iteration 1200 / 1500: loss 1.199192
# iteration 1300 / 1500: loss 1.155795
# iteration 1400 / 1500: loss 1.028442
# Training Accuracy: 0.613122 | Validation Accuracy: 0.506098
#
# Test accuracy:  0.5452436194895591

# LR: 7.134183e-02 | Reg: 5.196882e-05 | Train: 0.901018 | Val: 0.588415
# Training Accuracy: 0.998869 | Validation Accuracy: 0.585366
# Test accuracy:  0.5870069605568445


'''
iteration 0 / 1500: loss 1.791920
iteration 100 / 1500: loss 1.268280
iteration 200 / 1500: loss 1.170722
iteration 300 / 1500: loss 1.039042
iteration 400 / 1500: loss 0.979495
iteration 500 / 1500: loss 1.113674
iteration 600 / 1500: loss 1.018770
iteration 700 / 1500: loss 1.060840
iteration 800 / 1500: loss 1.052532
iteration 900 / 1500: loss 0.912105
iteration 1000 / 1500: loss 1.048558
iteration 1100 / 1500: loss 1.070518
iteration 1200 / 1500: loss 0.971529
iteration 1300 / 1500: loss 0.950717
iteration 1400 / 1500: loss 0.992034
Training Accuracy: 0.826357 | Validation Accuracy: 0.615854

Test accuracy:  0.6310904872389791
'''
# With data preprocessing
best_net = TwoLayerNet(input_size, 100, 6)
best_net.train(X_train, y_train, X_val, y_val,
              learning_rate=7.134183e-02, reg=5.196882e-02,
               num_iters=1500, verbose=True)

# might need to shuffle around the data division into training, test, and val tests

train_pred = best_net.predict(X_train)
train_acc = np.mean(train_pred == y_train)

val_pred = best_net.predict(X_val)
val_acc = np.mean(val_pred == y_val)

print("Training Accuracy: %f | Validation Accuracy: %f" % (train_acc, val_acc))

print()
# test_acc = (best_net.predict(X_test) == y_test).mean()
# print('Test accuracy: ', test_acc)

X_test_trainpro = (X_test - X_mean) / X_std
X_test_testpro = (X_test-np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
test_acc_trainpro = np.mean(best_net.predict(X_test_trainpro) == y_test)
test_acc_testpro = np.mean(best_net.predict(X_test_testpro) == y_test)
print("Test Accuracy with Train Statistics:", test_acc_trainpro)
print("Test Accuracy with Test Statistics:", test_acc_testpro)

print()
print("Test dataset statistics with train preprocessing.", np.mean(X_test_trainpro), np.std(X_test_trainpro))
print("Test dataset statistics with test preprocessing.", np.mean(X_test_testpro), np.std(X_test_testpro))

if test_acc_trainpro > test_acc_testpro:
    print("Train statistics work better.")
    # The two validation_accuracies should be fairly similar since the train and test datasets should have roughly the same statistics.
    # But the preprocessing using train statistics should still work a little better.

elif test_acc_trainpro < test_acc_testpro:
    print("Test statistics work better.")
    '''
    Even if it works better, you should still not use test statistics because...
    A) Calculating test statistics creates an additional step during testing_and_tools phase which slows down applications.
    B) The neural network might not always receive a large test dataset (maybe it is only classifying one or two
       images at a time). Then, the test dataset will be very noisy and data preprocessing will be essentially
       useless at that point.
    C) This one example where the train statistics perform worse could be an exception or an outlier.
        
    Logically speaking, the neural network will perform better when data preprocessing is done using
    the training data statistics because the same statistics is used. When the neural network trains, 
    it learns how to classify images using input data that was normalized/processed/changed based on the train statistics.
    But, if the test dataset is then normalized based on the test statistics, the neural network will receive
    different values since the inputs were normalized differently. The neural network will not know
    what to do with these different values and will likely perform worse. (The neural networks performs
    best when it the input statistics/distribution matches the values it saw during training.)   
    
    Data preprocessing using same training statistics ensures that each images is normalized the same and is on the same scale and
    has similar statistics and distributions. So each image will match the expected values the network saw during training
    since each image is standardized based on the same statistics. 
    
    Standardizing based on training statistics might cause the test inputs to not be unit gaussian with zero mean and unit variance.
    However, it is more important for the inputs to have the same values and distribution as the training than to have unit
    gaussian because unit gaussian is only really necessary for backpropagation and calculating gradients, and the neural
    network does not calculate gradients for the test data.
    
    '''

# images.py that was used to get the above validation_accuracies
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def get_images(data_set):
#     X = []
#     y = []
#     text_file_name = 'one-indexed-files-notrash_' + data_set + '.txt'
#     with open('all_the_images/' + text_file_name, 'r') as file:
#         all_images = file.readlines()
#
#     for im in all_images:
#         name, label = im.split(' ')
#         label = int(label[:-1]) - 1
#         material = ''
#         for char in name:
#             if char.isdigit():
#                 break
#             material += char
#
#         picture = Image.open('all_the_images/Garbage_classification/' + material + '/' + name)
#         picture = picture.resize((30, 30))
#         picture = np.array(picture).reshape(-1)
#         X.append(picture)
#         y.append(label)
#
#     return np.array(X), np.array(y)

