# Unwrapping everything to demonstrate your understanding
# 1.3 Both encoder and decoder use pytorch.rnn

import copy
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

from StepByStep import StepByStep

torch.manual_seed(23)

class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features

        """
        batch_first – If True, then the input and output tensors are provided as
        (batch, seq, feature) instead of (seq, batch, feature). Note that this does
        not apply to hidden or cell states. See the Inputs/Outputs sections below for
        details. Default: False

        Remember, the input to RNN is (128, 2, 2) {128 rows, 2 points, x and y}.
        This is in the format of (N, L, F)

        batch_first expects N, L, F
        batch_first=False expects L, N, F

        > Without batch_first, the hidden layer output is:
        tensor([[[-0.7281, -0.8194],
                [-0.9378, -0.5947]]], grad_fn=<StackBackward0>)

        This is because w/o batch_first, rnn thought L=128, N=2, F=2, meaning:
            2 rows, each with 128 points and each points with 2 features.
            Therefore, there will be 2 hidden states produced.

        > With batch_first=true, the hidden layer output is:
        tensor([[[ 0.4347, -0.0482],
            [-0.3260,  0.4595],
            [ 0.0828, -0.3325],
            ...
            (1 for each row)

        Here, the RNN treats our input as N=128, L=2, F=2 which is what we intended.

        N => batch size, aka how many rows in your data
        L => sequence length, aka how many points in each row
        F => number of features, aka how many x, y, z in each of your points
        """
        rnn = nn.RNN(input_size=2, hidden_size=2, num_layers=1, nonlinearity='tanh', batch_first=True)

        # Assign custom weights and biases
        with torch.no_grad():
            rnn.weight_ih_l0 = nn.Parameter(torch.tensor([
                [0.6627, -0.4245], [0.5373, 0.2294]
            ], dtype=torch.float64, requires_grad=True))

            rnn.bias_ih_l0 = nn.Parameter(torch.tensor([0.4954, 0.6533], dtype=torch.float64, requires_grad=True))

            rnn.weight_hh_l0 = nn.Parameter(torch.tensor([
                [-0.4015, -0.5385], [-0.1956, -0.6835]
            ], dtype=torch.float64, requires_grad=True))

            rnn.bias_hh_l0 = nn.Parameter(torch.tensor([-0.3565, -0.2904], dtype=torch.float64, requires_grad=True))

        self.rnn = rnn

    def forward(self, list_of_two_points):
        """Now batching. Returns a list of h1s

        NOTE: hidden is always initialized as zero, so we can do that here.
        """
        return self.rnn(list_of_two_points)


class Decoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.regression = nn.Linear(self.hidden_dim, self.n_features, dtype=torch.float64)
        self.rnn = nn.RNN(input_size=2, hidden_size=2, num_layers=1, nonlinearity='tanh', batch_first=True)

        with torch.no_grad():
            self.regression.weight.data = torch.tensor([
                [-0.1802, -0.3691],
                [-0.0514, 0.4502]], dtype=torch.float64)
            self.regression.bias.data = torch.tensor(
                [0.3566, -0.3189], dtype=torch.float64)

            self.rnn.weight_ih_l0 = nn.Parameter(torch.tensor([
                [0.6627, -0.4245], [ 0.5373,  0.2294]
            ], dtype=torch.float64, requires_grad=True))
            self.rnn.bias_ih_l0 = nn.Parameter(torch.tensor(
                [0.4954, 0.6533], dtype=torch.float64, requires_grad=True))
            # Hidden
            self.rnn.weight_hh_l0 = nn.Parameter(torch.tensor([
                [-0.4015, -0.5385], [-0.1956, -0.6835]
            ], dtype=torch.float64, requires_grad=True))
            self.rnn.bias_hh_l0 = nn.Parameter(torch.tensor(
                [-0.3565, -0.2904], dtype=torch.float64, requires_grad=True))

    def forward(self, point, hidden):
        """To use torch.RNN, we just package point and hidden into
        (N=1, L, F) with batch_first=True. This means each batch only has 1 row
        """
        # Refer to https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
        # on how squeezing and unsqueezing works

        # Since point is [x, y], we need to make it [ [x, y] ] as this shape has
        # (batch size N=1, sequence length L=1, feature number F=2)
        # before: tensor([ 0.8055, -0.9169])
        # after:  tensor([[ 0.8055, -0.9169]])
        point = torch.unsqueeze(point, 0)
        hidden = torch.unsqueeze(hidden, 0)
        useless, new_hidden = self.rnn(point, hidden)

        # > new_hidden tensor([[0.3081, 0.0360]], grad_fn=<SqueezeBackward1>)
        # Careful! new_hidden now has shape (1, 1, 2) but we want (1, 2)
        # again, this is because we batched them as N=1
        new_hidden = new_hidden.squeeze()

        new_point = self.regression(new_hidden)
        return new_point, new_hidden


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, input_len, target_len, teacher_forcing_prob=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_len = input_len
        self.target_len = target_len
        self.teacher_forcing_prob = teacher_forcing_prob
        self.outputs = None

    def init_outputs(self, batch_size):
        device = next(self.parameters()).device
        # N, L (target), F
        self.outputs = torch.zeros(batch_size,
                              self.target_len,
                              self.encoder.n_features).to(device)

    def store_output(self, i, out):
        # Stores the output
        self.outputs[:, i:i+1, :] = out

    def forward(self, X):
        """X is a tensor of shape N, 4, 2
        which means a list of N rows, each row has 4 points, each point have 2 dim
        """
        ## Encoder, now batching

        # no need to provide hidden, it will always be initialized to 0
        first_two_points = X[:, :2]      # all rows, index up to 2
        useless, h1s = self.encoder(first_two_points)

        ## Decoder, no batching yet
        y_hat = []

        for i, four_pts in enumerate(X):
            row_output = []
            input_pt = four_pts[1]      # starts off as 2nd point, then 3rd ...

            """
            h1s has 1 batch, 128 rows, 2 feature
            tensor([[[ 0.4347, -0.0482],
                [-0.3260,  0.4595],
                [ 0.0828, -0.3325],
                [-0.5562,  0.3439],
                ...
            """
            hidden = h1s[0][i]      # first batch, ith row

            for i in range(2):
                # Applies RNN and Regression
                output_point, new_hidden = self.decoder(input_pt, hidden)
                row_output.append(output_point)
                hidden, input_pt = new_hidden, output_point

            y_hat.append(row_output)

        """
        y_hat is now like this:
        [
            [
                tensor([
                    [ 0.2878, -0.3185]
                ], grad_fn=<AddmmBackward0>),
                tensor([
                    [ 0.4551, -0.3668]
                ], grad_fn=<ViewBackward0>)],
            ]

        """
        return y_hat




if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)

    encoder = Encoder(n_features=2, hidden_dim=2)
    decoder = Decoder(n_features=2, hidden_dim=2)
    model = EncoderDecoder(encoder, decoder, input_len=2, target_len=2, teacher_forcing_prob=0.5)
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    import pickle
    with open('random_data.pickle', 'rb') as inf:
        data = pickle.load(inf)

    # train_points and test_points each have 4 points
    train_points, train_directions = torch.tensor(data['points']), data['directions']
    test_points, test_directions = torch.tensor(data['test_points']), data['test_directions']

    # Shape of (128, 2, 2)
    train_last2_points = train_points[:, 2:]        # all rows, index 2 onwards
    test_last2_points = test_points[:, 2:]

    EPOCH = 20
    for epoch in range(EPOCH):
        model.train()

        # Now model takes in a list of 4 points
        y_hat = model(train_points)
        # y_hat is a list of 128 rows, each row has 2 points, each point has 2 dim

        # Convert y_hat to tensor (128, 2, 2)
        # First, ensure each pair is stacked to form a tensor of shape [2, 2]
        y_hat_pairs = [torch.stack(pair).squeeze(1) for pair in y_hat]  # This may require adjustment based on the exact shape of tensors in y_hat
        # Now, stack these pairs along a new dimension to get a tensor of shape [128, 2, 2]
        y_hat_tensor = torch.stack(y_hat_pairs)  # Resulting shape will be [128, 2, 2]
        training_loss = loss(y_hat_tensor, train_last2_points)

        training_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            y_hat = model(test_points)
            y_hat_pairs = [torch.stack(pair).squeeze(1) for pair in y_hat]  # This may require adjustment based on the exact shape of tensors in y_hat
            y_hat_tensor = torch.stack(y_hat_pairs)  # Resulting shape will be [128, 2, 2]
            testing_loss = loss(y_hat_tensor, test_last2_points)

        print(f'Epoch:{epoch} training:{training_loss:,.3f} validation:{testing_loss:,.3f}')
