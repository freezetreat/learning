# Using batches on GPU

##
## x = is a vector with 100 rows, y as well
## x is 100 x 1, y is also 100 x 1
##

import torch
import torch.optim as optim
from torchviz import make_dot
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

def step_fn(Y, X, W, B, loss_fn):
    """
    Y = W X + B

    returns loss
    """
    # Y.to('cuda')
    # X.to('cuda')

    Y_hat = torch.matmul(X, W.unsqueeze(1)) + B
    losses = Y_hat - Y

    # losses is an array. Need it to be a scalar to call backward.
    # note that for some reason, this differs from scratch.py because
    # there we use the sum of losses in our gradients.
    return loss_fn(losses)


loss_fn = lambda losses: (losses ** 2).mean()


for i in range(epoch):

    training_loss = step_fn(Y_train, X_train,
                            w_param, bias_param, loss_fn)

    # Note that we differ from the book's implementation because the book
    # includes backward in the compute_loss function. We don't so we have to
    # explicitly call it here.
    training_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 5 calculate losses
    with torch.no_grad():
        validation_loss = step_fn(Y_val, X_val,
                                  w_param, bias_param, loss_fn)

    logging.info(f'Epoch:{i} bias:{bias_param.item():,.3f} w0:{w_param[0].item():,.3f} w1:{w_param[1].item():,.3f} '
                #  f'bias_grad:{bias_grad:,.3f} w0_grad:{w0_grad:,.3f} w1_grad:{w1_grad:,.3f} '
                #  f'training:{training_loss:,.3f} ')
                 f'training:{training_loss:,.3f} validation:{validation_loss:,.3f}')


