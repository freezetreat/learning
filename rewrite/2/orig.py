
##
## import
##

import numpy as np
from sklearn.linear_model import LinearRegression

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('fivethirtyeight')

##
## data_generation/simple_linear_regression.py
## x = is a vector with 100 rows, y as well
##

import numpy as np

true_b = 1
true_w = 2
N = 100

# Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
y = true_b + true_w * x + (.1 * np.random.randn(N, 1))


##
## data_preparation/v2.py
##

torch.manual_seed(13)

x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# Contains all points, still in CPU
dataset = TensorDataset(x_tensor, y_tensor)

ratio = 0.8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train

train_data, val_data = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=16)


##
## Helper function #1, #2, #3
##

def make_train_step_fn(model, loss_fn, optimizer):
    """Builds a function that performs a step in the train loop, 1 epoch
    """
    def perform_train_step_fn(x, y):
        # Set the model to training mode
        model.train()
        # Get predicted output
        yhat = model(x)
        # Commpute loss
        loss = loss_fn(yhat, y)
        # Compute gradient for both parameters
        loss.backward()
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        # return the loss
        return loss.item()

    return perform_train_step_fn


def mini_batch(device, data_loader, step_fn):
    mini_batch_losses = []
    # Only when we do the actual computation we send it to GPU
    # GPU rams are expensive
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)

    loss = np.mean(mini_batch_losses)
    return loss


def make_val_step_fn(model, loss_fn, optimizer):

    def perform_val_step_fn(x, y):
        # Set the model to eval mode
        model.eval()
        # Get predicted output
        yhat = model(x)
        # Commpute loss
        loss = loss_fn(yhat, y)

        # No need for backward and optimizer since we don't want to update our parameters
        return loss.item()

    return perform_val_step_fn

##
## model_configuyration/v3.py
##

device = 'cuda'

lr = 0.1

torch.manual_seed(42)

model = nn.Sequential(nn.Linear(1, 1)).to(device)

optimizer = optim.SGD(model.parameters(), lr=lr)

loss_fn = nn.MSELoss(reduction='mean')

train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
val_step_fn = make_val_step_fn(model, loss_fn, optimizer)


##
## Tensor board stuff
##

writer = SummaryWriter('runs/test')


##
## Logging
##
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

##
## model_training/v4.py
##

n_epochs = 200

losses = []
val_losses = []

for epoch in range(n_epochs):

    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)

    # Validation!, no graidents in validation
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses.append(val_loss)

    if epoch % 10 == 0:
        logging.info(f'Epoch:{epoch}. Training:{loss:,.3f}. Validation:{val_loss:,.3f}')
        logging.info(f'Parameters:{[(k, v) for k, v in model.named_parameters()]}')
