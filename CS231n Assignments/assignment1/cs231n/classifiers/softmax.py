from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for train in range(num_train):
        scores = np.matmul(X[train, :], W)

        exp_scores = np.exp(scores)

        probabilities = exp_scores / np.sum(exp_scores)
        loss += -np.log(probabilities[y[train]])

        for score in range(num_classes):
            if score == y[train]:
                dW[:, score] += X[train] * (probabilities[score] - 1)
            else:
                dW[:, score] += X[train] * probabilities[score]

    dW /= num_train
    loss /= num_train

    loss += 0.5 * reg * np.sum(np.square(W))
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train, num_dims = X.shape
    num_classes = y.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.matmul(X, W)

    exps = np.exp(scores)
    totals = exps.sum(axis=1, keepdims=True)
    probs = exps/totals
    correct_probs = probs[range(num_train), y]
    loss += np.sum(-np.log(correct_probs))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W**2)

    dS = probs
    dS[range(num_train), y] -= 1
    dW = np.matmul(X.T, dS)
    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
