
##
## x = is a vector with 100 rows, y as well
## x is 100 x 1, y is also 100 x 1
##

import numpy as np

true_b = 1
true_w = 2
N = 100

# Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
y = true_b + true_w * x + (.1 * np.random.randn(N, 1))


## generate our training and validation sets

# Shuffles the indices, not needed since we have DataSet
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
# Uses the remaining indices for validation
val_idx = idx[int(N*.8):]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]


## Our formula is this
## y = bias + weight * x

# TODO initialize with pytorch(42) value
weight = 0
bias = 0


# NOTE:
# Loss must be MSE because we are implementing their formula


for idx, x in enumerate(x_train):
    # 1. compute yhat
    yhat = bias + weight * x
    # 2. compute loss
    loss = yhat - y_train[idx]

    # 3. compute gradient descent
    # 4. update parameters
    # 5. repeat



