import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

SEED = 23
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

### Load data - make sure you copy this EXACTLY as zero_copy
with open('random_data.pickle', 'rb') as inf:
    data = pickle.load(inf)

points, directions = data['points'], data['directions']
full_train = torch.as_tensor(points).float()
target_train = full_train[:, 2:]
test_points, test_directions = data['test_points'], data['test_directions']
full_test = torch.as_tensor(test_points).float()
source_test = full_test[:, :2]
target_test = full_test[:, 2:]

generator = torch.Generator()

train_data = TensorDataset(full_train, target_train)
test_data = TensorDataset(source_test, target_test)
train_loader = DataLoader(train_data, batch_size=16, shuffle=False, generator=generator)
test_loader = DataLoader(test_data, batch_size=16)


# Parameters
input_size = 2
hidden_size = 2
output_size = 2

# Define the GRU and Linear layers directly
# gru_layer = nn.GRU(input_size, hidden_size, batch_first=True)
# linear_layer = nn.Linear(hidden_size, output_size)

learning_rate = 0.01
n_epochs = 10

loss_fn = nn.MSELoss()

device = 'cpu'



from zero_copy import Encoder, Decoder, EncoderDecoder
encoder = Encoder(n_features=2, hidden_dim=2)
decoder = Decoder(n_features=2, hidden_dim=2)
model = EncoderDecoder(encoder, decoder, input_len=2, target_len=2, teacher_forcing_prob=1.0)

# Optimizer (note: it's necessary to pass the parameters of both layers)
#optimizer = optim.Adam(list(gru_layer.parameters()) + list(linear_layer.parameters()), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=0.01)


"""
Here, we will attempt to re-create 0_copy by unwrapping everything
"""


def train_step_fn(x, y):
    """Takes a batch of x and y and returns the loss
    """
    # TODO manually turn this off
    model.train()       # Set the model to TRAIN mode

    # Forward pass - Compute predicted output
    yhat = model(x)
    # Compute Loss
    loss = loss_fn(yhat, y)
    # Compute gradients
    loss.backward()

    # Update parameters
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def val_step_fn(x, y):
    """Takes a batch of x and y and returns the loss
    """
    # TODO - manually turn this off
    model.eval()        # set the model to validation mode

    yhat = model(x)
    loss = loss_fn(yhat, y)
    return loss.item()


def _mini_batch(validation=False):
    if validation:
        data_loader = test_loader
        step_fn = val_step_fn
    else:
        data_loader = train_loader
        step_fn = train_step_fn

    n_batches = len(data_loader)
    mini_batch_losses = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)

    loss = np.mean(mini_batch_losses)
    return loss


total_epoch = 0
losses = []

for epoch in range(n_epochs):
    total_epoch += 1

    loss = _mini_batch(validation=False)
    losses.append(loss)

    with torch.no_grad():       # Validation
        val_loss = _mini_batch(validation=True)
        print(epoch, val_loss)

    # print(list(model.parameters()))


