# Using nn.module


import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


np.random.seed(42)

import logging
logging.basicConfig(format='%(message)s', level=logging.DEBUG)



# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Copy

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



class Basic(nn.Module):
    def __init__(self):
        super().__init__()
        # Copied directly from 3_tensor
        self.w = nn.Parameter(
            torch.tensor([0, 0], dtype=torch.float64, requires_grad=True))
        self.b = nn.Parameter(
            torch.tensor([0], dtype=torch.float64, requires_grad=True))

    def forward(self, x):
        # Copied directly from 3_tensor as well
        return torch.matmul(x, self.w.unsqueeze(1)) + self.b



epoch = 20
lr = 0.1

model = Basic()

# Copied directly from 4_optimizer
optimizer = optim.SGD(model.parameters(), lr=lr)


for i in range(epoch):

    model.train()       # Set in training mode
    Y_hat = model(X_train)
    losses = Y_hat - Y_train
    training_loss = (losses ** 2).mean()    # Copied directly from 4_optimizer

    # Back propagation
    training_loss.backward()

    # Copied from 4_directly
    optimizer.step()
    optimizer.zero_grad()

   # calculate losses
    with torch.no_grad():
        model.eval()
        Y_hat = model(X_val)
        losses = Y_hat - Y_val
        validation_loss = (losses ** 2).mean()    # Copied directly from 4_optimizer

    param = model.state_dict()
    logging.info(f'Epoch:{i} bias:{param['b'].item():,.3f} w0:{param['w'][0].item():,.3f} w1:{param['w'][1].item():,.3f} '
                 f'training:{training_loss:,.3f} validation:{validation_loss:,.3f}')
