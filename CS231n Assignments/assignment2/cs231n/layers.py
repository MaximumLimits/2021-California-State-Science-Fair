from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = x.reshape((x.shape[0], -1)).dot(w)
    out += b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    input_shape = x.shape

    x = x.reshape((x.shape[0], -1))

    dx = dout.dot(w.T).reshape(input_shape)
    dw = x.T.dot(dout)
    db = np.sum(dout, axis=0)

    # num = dout.shape[0]
    # dx /= num
    # dw /= num
    # db /= num

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = (x > 0).astype(float)
    dx *= dout

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    layernorm = bn_param.get('layernorm', False)
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mean = np.mean(x, axis=0)
        variance = np.var(x, axis=0) + eps
        standard_deviation = np.sqrt(variance)

        norm_x = (x - mean)/standard_deviation
        out = norm_x*gamma + beta

        if not layernorm:
            running_mean = momentum * running_mean + (1-momentum) * mean
            running_var = momentum * running_var + (1-momentum) * variance

        axis = 1 if layernorm else 0
        cache = (x, mean, variance, standard_deviation, norm_x, gamma, beta, eps, axis)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        norm_x = (x - running_mean) / (np.sqrt(running_var + eps))
        out = norm_x * gamma + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, mean, variance, std, norm_x, gamma, beta, eps, axis = cache
    N = x.shape[0]

    dgamma = np.sum(norm_x * dout, axis=axis)
    dbeta = np.sum(dout, axis=axis)

    x_minus_mean = x - mean
    dnorm_x = gamma * dout
    dx_minus_mean = dnorm_x / std
    dx_inverse_std = np.sum(dnorm_x * x_minus_mean, axis=0)
    dxstd = dx_inverse_std * -1 / (std ** 2)
    dvar = dxstd * 0.5 * 1 / std
    dx_minus_mean_squared = dvar / N
    dx_minus_mean2 = dx_minus_mean_squared * 2 * x_minus_mean
    dx1 = (dx_minus_mean2 + dx_minus_mean)
    dmean = np.sum((dx_minus_mean2 + dx_minus_mean) * -1, axis=0)
    dx2 = dmean / N
    dx = dx1 + dx2


    # dbeta = np.sum(dout, axis=0)
    # dgamma = np.sum(norm_x * dout, axis=0)
    # d9 = gamma * dout
    # d8 = np.sum(x_minus_mean * d9, axis=0)
    # d31 = d9/std
    # d7 = -d8/(std**2)
    # d6 = d7/(2*std)
    # d5 = d6/N
    # d4 = d5
    # d32 = 2 * x_minus_mean * d4
    # d3 = d31 + d32
    # d11 = d3
    # d2 = np.sum(-d3, axis=0)
    # d12 = d2/N
    # dx = d11 + d12

    # N = 1.0 * dout.shape[0]
    # dfdz = dout * gamma  # [NxD]
    # dudx = 1 / N  # [NxD]
    # dvdx = 2 / N * (x - mean)  # [NxD]
    # dzdx = 1 / std  # [NxD]
    # dzdu = -1 / std  # [1xD]
    # dzdv = -0.5 * (variance ** -1.5) * (x - mean)  # [NxD]
    # dvdu = -2 / N * np.sum(x - mean, axis=0)  # [1xD]
    #
    # dx = dfdz * dzdx + np.sum(dfdz * dzdu, axis=0) * dudx + \
    #      np.sum(dfdz * dzdv, axis=0) * (dvdx + dvdu * dudx)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, mean, variance, std, norm_x, gamma, beta, eps, axis = cache
    dgamma = np.sum(norm_x * dout, axis=axis)
    dbeta = np.sum(dout, axis=axis)

    N = dout.shape[0]

    z = norm_x
    dfdz = dout * gamma  # [NxD]
    dfdz_sum = np.sum(dfdz, axis=0)  # [1xD]
    dx = dfdz - dfdz_sum / N - np.sum(dfdz * z, axis=0) * z / N  # [NxD]
    dx /= std

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ln_param['mode'] = 'train'
    ln_param['layernorm'] = True

    out, cache = batchnorm_forward(x.T, gamma.reshape(-1, 1), beta.reshape(-1, 1), ln_param)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out.T, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx, dgamma, dbeta = batchnorm_backward_alt(dout.T, cache)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx.T, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = np.random.random(x.shape) < p
        out = x * mask / p

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        ddropout = mask / dropout_param['p']
        dx = ddropout * dout

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    assert (H + 2*pad - HH) % stride == 0, 'Height dimensions do not work.'
    assert (W + 2 * pad - WW) % stride == 0, 'Width dimensions do not work.'

    out_shape = int((H + 2 * pad - HH) / stride) + 1  # technically this will only work if the input and filters are squares
    out = np.zeros((N, F, out_shape, out_shape))

    # for ixtrain, train in enumerate(x[:]):  # probably should not be calling the values 'ixtrain' and 'train' since this layer will act the same during testing
    #     pad_x = np.zeros((train.shape[0], train.shape[1] + 2 * pad, train.shape[2] + 2 * pad))
    #     for grid in range(len(pad_x)):
    #         pad_x[grid] = np.pad(x[ixtrain, grid], 1, mode='constant')
    #
    #     # loop through each filter (f is one filter) and each filter bias together (b is one bias)
    #     for f, bias, findex in zip(w[:], b, range(F)):
    #         c_out = np.zeros((out_shape, out_shape))  # new c_out for each filter
    #         for row_pos in range(0, pad_x.shape[1]-HH+1, stride):
    #             for col_pos in range(0, pad_x.shape[2]-WW+1, stride):
    #                 # r_field = np.array([pad_x[channel, row, col] for channel in range(C) for row in range(row_pos, row_pos + HH) for col in range(col_pos, col_pos + WW)])
    #                 r_field = np.array([pad_x[channel, row_pos:row_pos+HH, col_pos:col_pos+WW] for channel in range(C)])
    #                 r_field = r_field.reshape((C, HH, WW))
    #                 neuron = np.sum(r_field * f) + bias
    #                 c_out[int(row_pos/stride), int(col_pos/stride)] = neuron
    #         out[ixtrain, findex, :, :] = c_out

    w_row = w.reshape(F, -1)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    PH = x_pad.shape[2]
    PW = x_pad.shape[3]

    neuron = 0
    x_col = np.zeros((N, HH*WW*C, out_shape*out_shape))
    for row in range(0, PH-HH+1, stride):
        for col in range(0, PW-WW+1, stride):
            r_field = x_pad[:, :, row:row+HH, col:col+WW]
            x_col[:, :, neuron] = r_field.reshape(N, -1)
            neuron += 1
    out = (np.dot(w_row, x_col) + b[:, np.newaxis, np.newaxis]).transpose(1, 0, 2).reshape((N, F, out_shape, out_shape))

    # w_row = w.reshape(F, -1)
    # x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    # out_shape = int((H + 2 * pad - HH) / stride) + 1  # technically this will only work if the input and filters are squares
    # PH = x_pad.shape[2]
    # PW = x_pad.shape[3]
    #
    # x_col = np.zeros((N, C * HH * WW, out_shape * out_shape))
    # neuron = 0
    # for row in range(0, PH - HH + 1, stride):
    #     for col in range(0, PW - WW + 1, stride):
    #         r_field = x_pad[:, :, row:row + HH, col:col + WW].reshape(N, -1)
    #         x_col[:, :, neuron] = r_field
    #         neuron += 1
    # out = (np.dot(w_row, x_col) + b.reshape(F, 1, 1)).transpose(1, 0, 2).reshape(N, F, out_shape, out_shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, w, b, conv_param = cache

    db = np.sum(dout, axis=(0, 2, 3))

    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    # first successful (but very naive) attempt
    # for ixtrain, train in enumerate(x[:]):
    #     pad_h = train.shape[1] + 2 * pad
    #     pad_w = train.shape[2] + 2 * pad
    #     pad_x = np.zeros((train.shape[0], pad_h, pad_w))
    #     for grid in range(len(pad_x)):
    #         pad_x[grid] = np.pad(x[ixtrain, grid], 1, mode='constant')
    #     d0p = np.zeros_like(pad_x)
    #
    #     # for f, findex in zip(w[:], range(F)):
    #     #     for row_pos in range(0, pad_x.shape[1] - HH + 1, stride):
    #     #         for col_pos in range(0, pad_x.shape[2] - WW + 1, stride):
    #     #             r_field = np.array([pad_x[channel, row_pos:row_pos + HH, col_pos:col_pos + WW] for channel in range(C)])
    #     #             r_field = r_field.reshape((C, HH, WW))
    #     #             r_field_dout = dout[ixtrain, findex, int(row_pos / stride), int(col_pos / stride)]
    #     #             dw[findex] += r_field * r_field_dout
    #
    #     for f, findex in zip(w[:], range(F)):
    #         for row_pos in range(0, pad_x.shape[1] - HH + 1, stride):
    #             for col_pos in range(0, pad_x.shape[2] - WW + 1, stride):
    #                 r_field = np.array([pad_x[channel, row_pos:row_pos + HH, col_pos:col_pos + WW] for channel in range(C)])
    #                 r_field = r_field.reshape((C, HH, WW))
    #                 r_field_dout = dout[ixtrain, findex, int(row_pos / stride), int(col_pos / stride)]
    #                 dw[findex] += r_field * r_field_dout
    #                 d0p[:, row_pos:row_pos + HH, col_pos:col_pos + WW] += f * r_field_dout
    #
    #     dno0p = np.delete(d0p, [range(pad), range(pad_h-1, pad_h-1+pad)], axis=1)
    #     dno0p = np.delete(dno0p, [range(pad), range(pad_w-1, pad_w-1+pad)], axis=2)
    #     dx[ixtrain] += dno0p

    # failed attempt to do backward pass without looping through each input example N
    # w_row = w.reshape(F, -1)
    # x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    # out_shape = int((H + 2 * pad - HH) / stride) + 1  # technically this will only work if the input and filters are squares
    # PH = x_pad.shape[2]
    # PW = x_pad.shape[3]
    #
    # x_col = np.zeros((N, C*HH*WW, out_shape*out_shape))
    # neuron = 0
    # for row in range(0, PH-HH+1, stride):
    #     for col in range(0, PW-WW+1, stride):
    #         r_field = x_pad[:, :, row:row+HH, col:col+WW].reshape(N, -1)
    #         x_col[:, :, neuron] = r_field
    #         neuron += 1
    #
    # # dw = np.dot(dout.reshape(N, F, out_shape*out_shape), x_col.transpose((0, 2, 1))).reshape((F, C, HH, WW))
    # dw = np.sum(np.dot(dout.reshape(N, F, out_shape*out_shape), x_col.transpose((0, 2, 1))).transpose((0, 2, 1, 3)), axis=(0, 1)).reshape((F, C, HH, WW))
    # print(dw.shape)
    # print(F, C, HH, WW)

    # w_row = w.reshape(F, -1)
    # x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    # dx_pad = np.zeros_like(x_pad)
    # out_shape = int((H + 2 * pad - HH) / stride) + 1  # technically this will only work if the input and filters are squares
    # PH = x_pad.shape[2]
    # PW = x_pad.shape[3]
    # x_col = np.zeros((C * HH * WW, out_shape * out_shape))
    #
    # for index in range(N):
    #     dout_col = dout[index].reshape(F, out_shape * out_shape)
    #     neuron = 0
    #     for row in range(0, PH-HH+1, stride):
    #         for col in range(0, PW-WW+1, stride):
    #             r_field = x_pad[index, :, row:row+HH, col:col+WW].reshape(-1)
    #             x_col[:, neuron] = r_field
    #             dx_pad[index, :, row:row+HH, col:col+WW] += np.sum(w_row * dout_col[:, neuron][:, np.newaxis], axis=0).reshape((C, HH, WW))
    #             neuron += 1
    #
    #     dw += np.dot(dout[index].reshape(F, out_shape*out_shape), x_col.T).reshape((F, C, HH, WW))
    # dx = dx_pad[:, :, pad:-pad, pad:-pad]

    w_row = w.reshape(F, -1)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out_shape = int((H + 2 * pad - HH) / stride) + 1  # technically this will only work if the input and filters are squares
    PH = x_pad.shape[2]
    PW = x_pad.shape[3]

    x_col = np.zeros((C * HH * WW, out_shape * out_shape))

    for index in range(N):
        dout_col = dout[index].reshape(F, out_shape * out_shape)
        dx_pad_col = np.dot(w_row.T, dout_col)  # C*HH*WW, out_shape*out_shape
        dx_pad = np.zeros((C, PH, PW))
        neuron = 0
        for row in range(0, PH-HH+1, stride):
            for col in range(0, PW-WW+1, stride):
                r_field = x_pad[index, :, row:row+HH, col:col+WW].reshape(-1)
                x_col[:, neuron] = r_field
                dx_pad[:, row:row+HH, col:col+WW] += dx_pad_col[:, neuron].reshape((C, HH, WW))
                neuron += 1

        dw += np.dot(dout_col, x_col.T).reshape((F, C, HH, WW))
        dx[index] = dx_pad[:, pad:-pad, pad:-pad]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    N, C, H, W = x.shape

    assert (H + - pool_h) % stride == 0, 'Height dimensions do not work.'
    assert (W + - pool_w) % stride == 0, 'Width dimensions do not work.'

    out_shape = int((H - pool_h) / stride) + 1  # technically this will only work if the input and filters are squares
    out = np.zeros((N, C, out_shape, out_shape))

    for ixtrain, train in enumerate(x[:]):  # probably should not be calling the values 'ixtrain' and 'train' since this layer will act the same during testing
        for row_pos in range(0, x.shape[2] - pool_h + 1, stride):
            for col_pos in range(0, x.shape[3] - pool_w + 1, stride):
                r_field = np.array([x[ixtrain, channel, row_pos:row_pos + pool_h, col_pos:col_pos + pool_h] for channel in range(C)])
                r_field = r_field.reshape((C, pool_h, pool_w))
                neurons = np.max(r_field, axis=(1, 2))
                out[ixtrain, :, int(row_pos/stride), int(col_pos/stride)] = neurons

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, pool_param = cache

    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    N, C, H, W = x.shape

    dx = np.zeros_like(x)
    for ixtrain, train in enumerate(x[:]):  # probably should not be calling the values 'ixtrain' and 'train' since this layer will act the same during testing
        for row_pos in range(0, x.shape[2] - pool_h + 1, stride):
            for col_pos in range(0, x.shape[3] - pool_w + 1, stride):
                r_field = np.array([x[ixtrain, channel, row_pos:row_pos + pool_h, col_pos:col_pos + pool_h] for channel in range(C)])
                r_field = r_field.reshape((C, pool_h, pool_w))
                neurons = np.max(r_field, axis=(1, 2))
                for channel in range(C):
                    row, col = np.nonzero(r_field[channel] == neurons[channel])
                    x_row = row + row_pos
                    x_col = col + col_pos
                    dx[ixtrain, channel, x_row, x_col] += dout[ixtrain, channel, int(row_pos/stride), int(col_pos/stride)]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    x_reshape = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out, cache = batchnorm_forward(x_reshape, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dout_reshape = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout_reshape, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    assert C % G == 0, 'G is not a divisor of C'

    x_reshape = x.reshape(N, -1)
    x_group = x_reshape.reshape(-1, int(C / G)).T

    mean = np.mean(x_group, axis=0, keepdims=True)
    variance = np.var(x_group, axis=0, keepdims=True) + eps
    standard_deviation = np.sqrt(variance)

    norm_x = (x_group - mean) / standard_deviation
    norm_x = norm_x.T.reshape(N, C, H, W)
    gamma = gamma.reshape(1, C, 1, 1)
    beta = beta.reshape(1, C, 1, 1)
    out = norm_x * gamma + beta

    cache = (x_group, mean, variance, standard_deviation, norm_x, gamma, beta, eps, G)

    # x_reshape = x.transpose(0, 2, 3, 1).reshape(-1, C)
    # x_group = x_reshape.reshape(-1, int(C/G))
    # gamma_reshape = np.tile(gamma.reshape(-1, int(C/G)), (N*H*W, 1))
    # beta_reshape = np.tile(beta.reshape(-1, int(C/G)), (N*H*W, 1))
    #
    # mean = np.mean(x_group, axis=1, keepdims=True)
    # variance = np.var(x_group, axis=1, keepdims=True) + eps
    # standard_deviation = np.sqrt(variance)
    #
    # norm_x = (x_group - mean) / standard_deviation
    # out = norm_x * gamma_reshape + beta_reshape
    #
    # out = out.reshape(-1, C).reshape(N, H, W, C).transpose(0, 3, 1, 2)
    #
    # axis = 1
    # cache = (x_group, mean, variance, standard_deviation, norm_x, gamma, beta, eps, G)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape

    x, mean, variance, std, norm_x, gamma, beta, eps = cache

    dgamma = np.sum(norm_x * dout, axis=1)
    dbeta = np.sum(dout, axis=1)

    N = dout.shape[0]
    z = norm_x
    dfdz = dout * gamma  # [NxD]
    dfdz_sum = np.sum(dfdz, axis=0)  # [1xD]
    dx = dfdz - dfdz_sum / N - np.sum(dfdz * z, axis=0) * z / N  # [NxD]
    dx /= std

    dx = dx.reshape(-1, C).reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma , dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

    # N = x.shape[0]
    # shifted_logits = x - np.max(x, axis=1, keepdims=True)
    # exps = np.exp(shifted_logits)
    # probs = exps/np.sum(exps, axis=1, keepdims=True)
    # correct_probs = probs[range(N), y]
    # loss = np.sum(-np.log(correct_probs)) / N
    # dx = probs.copy()
    # dx[range(N), y] -= 1
    # dx /= N
    # return loss, dx