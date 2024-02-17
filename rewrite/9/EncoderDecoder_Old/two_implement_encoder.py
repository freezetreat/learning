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

"""
TODO: unwrap encoder, decoder and EncoderDecoder
"""

class Encoder(object):
    """Not subclassing nn.Module so we can implement the logic ourselves
    """
    def __init__(self, n_features=2, hidden_dim=2):
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.hidden = None
        self.basic_rnn = nn.GRU(self.n_features, self.hidden_dim, batch_first=True)

    # 1st: resolve `hidden_seq = self.encoder(source_seq)`
    def __call__(self, source_seq):
        """Takes in the following a sequence of points:

        tensor([[[ 1.0349,  0.9661],
                [ 0.8055, -0.9169]],

                [[ 1.0185, -1.0651],
                [ 0.8879,  0.9653]],

                [[-1.0292,  1.0127],
                [-1.1247, -0.9683]],

                [[-1.0341, -0.8910],
                [-0.9549,  0.9506]],

                [[-0.9856,  1.0905],
                [-0.9599, -1.0765]],

        and outputs a sequence of hidden states.

        This is how you use nn.GRU
        >>> rnn = nn.GRU(10, 20, 2)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
        """
        init_hidden = torch.tensor([0, 0])

        h_1s = []
        for first_second in source_seq:
            first, second = first_second

            throwaway, h_0 = self.basic_rnn(first, init_hidden)
            throwaway, h_1 = self.basic_rnn(second, h_0)
            h_1s.append(h_1)

        return torch.tensor([torch.tensor(x) for x in h_1s])


from zero_copy import Decoder, EncoderDecoder
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


