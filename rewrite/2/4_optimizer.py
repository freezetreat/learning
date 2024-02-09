# Using optimizer

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

import torch
import torch.optim as optim
import numpy as np

# Data Generation
np.random.seed(42)


true_b = 1
true_w1 = 2
true_w2 = -0.5
N_orig = 100

# Y = B + W * X
B = np.full((N_orig, 1), true_b)         # N x 1
X = np.random.rand(N_orig, 2)            # N x 2
W = np.array([true_w1, true_w2])         # 1 * 2
Y = (X * W).sum(axis=1, keepdims=True) + B      # N * 1

#noise = np.random.rand(N_orig, 2) * .1
#Y = B + np.sum(W * (X + noise), axis=1, keepdims=True)


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

# New: we are using pytorch here!
X_train, Y_train = torch.from_numpy(X_train), torch.from_numpy(Y_train)
X_val, Y_val = torch.from_numpy(X_val), torch.from_numpy(Y_val)

##
##

import logging
logging.basicConfig(format='%(message)s', level=logging.DEBUG)


# NOTE: W_param and bias_param are just the same values being repeated to N times
# it's to make computation easier
w_param = torch.tensor([0, 0], dtype=torch.float64, requires_grad=True)
bias_param = torch.tensor([0], dtype=torch.float64, requires_grad=True)


# NOTE:
# Loss must be MSE because we are implementing their formula

optimizer = optim.SGD([w_param, bias_param],
                      lr=0.1)

epoch = 20
# epoch = 5

for i in range(epoch):
    # Compute loss
    Y_hat = torch.matmul(X_train, w_param.unsqueeze(1)) + bias_param

    losses = Y_hat - Y_train

    # losses is an array. Need it to be a scalar to call backward.
    # note that for some reason, this differs from scratch.py because
    # there we use the sum of losses in our gradients.
    training_loss = (losses ** 2).mean()

    # now you can call backward since total_loss is a scalar
    training_loss.backward()

    # GPT: After losses.backward() is executed, the gradients are stored in the .grad
    # attribute of the tensors (w_param.grad and bias_param.grad).

    # Note, normally we replace with this:
    optimizer.step()
    optimizer.zero_grad()

    # 5 calculate losses
    with torch.no_grad():
        Y_hat = torch.matmul(X_val, w_param.unsqueeze(1)) + bias_param
        temp_losses = Y_hat - Y_val
        validation_loss = (temp_losses ** 2).mean()

    logging.info(f'Epoch:{i} bias:{bias_param.item():,.3f} w0:{w_param[0].item():,.3f} w1:{w_param[1].item():,.3f} '
                #  f'bias_grad:{bias_grad:,.3f} w0_grad:{w0_grad:,.3f} w1_grad:{w1_grad:,.3f} '
                #  f'training:{training_loss:,.3f} ')
                 f'training:{training_loss:,.3f} validation:{validation_loss:,.3f}')


