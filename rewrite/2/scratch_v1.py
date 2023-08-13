
# Road Map
# - use GPU

# Questions
# - why does optim.SGD requires model.parameters?
# - when evaluating validation data, why do we have to specify no_grad?
# - what is optimizer.zero_grad for?


##
## x = is a vector with 100 rows, y as well
## x is 100 x 1, y is also 100 x 1
##

import numpy as np

# Data Generation
np.random.seed(42)


true_b = 1
true_w1 = 2
true_w2 = -0.5
N_orig = 100

B = np.full((N_orig, 1), true_b)         # N x 1
X = np.random.rand(N_orig, 2)            # N x 2
W = np.full((N_orig, 2), (true_w1, true_w2))     # N x 2
# noise = np.random.rand(N_orig, 2) * .1
noise = np.zeros((N_orig, 2))
Y = B + np.sum(W * (X + noise), axis=1, keepdims=True)

# y = true_b + \
#     true_w1 * x1 + (.1 * np.random.randn(N, 1)) + \
#     true_w2 * x2 + (.1 * np.random.randn(N, 1))


## generate our training and validation sets

# Shuffles the indices, not needed since we have DataSet
idx = np.arange(N_orig)
np.random.shuffle(idx)

N = int(N_orig*.8)

# Uses first 80 random indices for train
train_idx = idx[:N]
# Uses the remaining indices for validation
val_idx = idx[N:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]

##
##

import logging
logging.basicConfig(format='%(message)s', level=logging.DEBUG)

## Our formula is this
## y = bias + weight * x

# TODO initialize with pytorch(42) value

# NOTE: W_param and bias_param are just the same values being repeated to N times
# it's to make computation easier
W_param = np.zeros((N, 2))
bias_param = np.zeros((N, 1))


# NOTE:
# Loss must be MSE because we are implementing their formula

epoch = 1000

for i in range(epoch):
    Y_hat = bias_param + np.sum(W_param * X_train, axis=1, keepdims=True)
    losses = Y_hat - Y_train            # N x 1

    # 3. compute gradient descent
    # d_MSE / d_b
    bias_grad = 2.0 * 1 / N * losses.sum()
    # d_MSE / d_w
    # first column of X which is x_1
    w0_grad = 2.0 * 1 / N * (X_train[:, 0] * losses.flatten()).sum()
    w1_grad = 2.0 * 1 / N * (X_train[:, 1] * losses.flatten()).sum()

    # 4. update parameters
    lr = 0.1
    bias_param -= lr * bias_grad
    # print('before', W_param)
    W_param[:, 0] -= lr * w0_grad
    W_param[:, 1] -= lr * w1_grad
    # print('after', W_param)

    # 5 calculate losses
    training_loss = 1 / N * (losses**2).sum()

    validation_losses = []
    # Reshape W_param and bias_param accordingly
    W_param_val = np.full((N_orig - N, 2), W_param[0])
    bias_param_val = np.full((N_orig - N, 1), bias_param[0])

    Y_hat_val = bias_param_val + np.sum(W_param_val * X_val, axis=1, keepdims=True)
    val_losses = Y_hat_val - Y_val
    val_loss = 1 / N * (val_losses**2).sum()

    logging.info(f'Epoch:{i} bias:{bias_param[0][0]:,.3f} w0:{W_param[0][0]:,.3f} w1:{W_param[0][1]:,.3f} '
                 f'bias_grad:{bias_grad:,.3f} w0_grad:{w0_grad:,.3f} w1_grad:{w1_grad:,.3f} '
                 f'training:{training_loss:,.3f} validation:{val_loss:,.3f}')


