# Everything using raw numbers and variables, no matrices whatsoevers

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

true_b = 1
true_w0 = 2
true_w1 = -0.5
N = 100

# Data Generation
np.random.seed(42)

# Same code as scratch_v1
X = np.random.rand(N, 2)            # N x 2
x1 = X[:, 0]
x2 = X[:, 1]

y = true_b + \
    true_w0 * x1 + \
    true_w1 * x2
# y = true_b + \
#     true_w0 * x1 + (.1 * np.random.randn(N, 1)) + \
#     true_w1 * x2 + (.1 * np.random.randn(N, 1))


## generate our training and validation sets

# Shuffles the indices, not needed since we have DataSet
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
# Uses the remaining indices for validation
val_idx = idx[int(N*.8):]

# Generates train and validation sets
x1_train, x2_train, y_train = x1[train_idx], x2[train_idx], y[train_idx]
x1_val, x2_val, y_val = x1[val_idx], x2[val_idx], y[val_idx]

x1_train, x2_train, y_train = x1_train.flatten(), x2_train.flatten(), y_train.flatten()
x1_val, x2_val, y_val = x1_val.flatten(), x2_val.flatten(), y_val.flatten()


##
##


import logging
logging.basicConfig(format='%(message)s', level=logging.DEBUG)



## Our formula is this
## y = bias + weight * x

# TODO initialize with pytorch(42) value
w0 = 0
w1 = 0
bias = 0


# NOTE:
# Loss must be MSE because we are implementing their formula

n = len(x1_train)
epoch = 10

for i in range(epoch):
    losses = []
    for idx in range(len(x1_train)):
        x1, x2 = x1_train[idx], x2_train[idx]
        # 1. compute yhat
        yhat = bias + w0 * x1 + w1 * x2
        # 2. compute loss
        loss = yhat - y_train[idx]
        losses.append(loss)

    # 3. compute gradient descent
    # d_MSE / d_b
    bias_grad = 2.0 * 1 / n * sum(losses)
    # d_MSE / d_w
    w0_grad = 2.0 * 1 / n * sum([x * losses[i] for (i, x) in enumerate(x1_train)])
    w1_grad = 2.0 * 1 / n * sum([x * losses[i] for (i, x) in enumerate(x2_train)])

    # 4. update parameters
    lr = 0.1
    bias -= lr * bias_grad
    w0 -= lr * w0_grad
    w1 -= lr * w1_grad

    # 5 calculate losses
    training_loss = 1 / n * sum([loss**2 for loss in losses])

    validation_losses = []
    for idx in range(len(x1_val)):
        x1, x2 = x1_val[idx], x2_val[idx]
        yhat = bias + w0 * x1 + w1 * x2
        loss = yhat - y_train[idx]
        validation_losses.append(loss)
    validation_loss = 1 / n * sum([loss**2 for loss in validation_losses])

    logging.info(f'Epoch:{i} bias:{bias:,.3f} w0:{w0:,.3f} w1:{w1:,.3f} '
                 f'bias_grad:{bias_grad:,.3f} w0_grad:{w0_grad:,.3f} w1_grad:{w1_grad:,.3f} '
                 f'training_loss:{training_loss:,.3f} validation_loss:{validation_loss:,.3f}')

