# Unwrapping everything to demonstrate your understanding
# 1.0 Naive approach, one by one (no batching)

import copy
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

from StepByStep import StepByStep


class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features

        self.weight_ih = nn.Parameter(torch.tensor([
            [0.6627, -0.4245], [ 0.5373,  0.2294]
        ], dtype=torch.float64, requires_grad=True))
        self.bias_ih = nn.Parameter(torch.tensor([0.4954, 0.6533], dtype=torch.float64, requires_grad=True))

        # Hidden
        self.weight_hh = nn.Parameter(torch.tensor([
            [-0.4015, -0.5385], [-0.1956, -0.6835]
        ], dtype=torch.float64, requires_grad=True))
        self.bias_hh = nn.Parameter(torch.tensor([-0.3565, -0.2904], dtype=torch.float64, requires_grad=True))

    def forward(self, point, hidden):
        """Takes in one at a time
        """
        #output, new_hidden = self.basic_rnn(point)
        x_cord, y_cord = point

        ix = x_cord * self.weight_ih[0][0] \
            + y_cord * self.weight_ih[0][1] \
            + self.bias_ih[0]

        iy = x_cord * self.weight_ih[1][0] \
            + y_cord * self.weight_ih[1][1] \
            + self.bias_ih[1]

        hx = hidden[0] * self.weight_hh[0][0] \
            + hidden[1] * self.weight_hh[0][1] \
            + self.bias_hh[0]

        hy = hidden[0] * self.weight_hh[1][0] \
            + hidden[1] * self.weight_hh[1][1] \
            + self.bias_hh[1]

        pretan_x = ix + hx
        pretan_y = iy + hy
        new_hidden = torch.stack([torch.tanh(pretan_x), torch.tanh(pretan_y)])

        return None, new_hidden


class Decoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.regression = nn.Linear(self.hidden_dim, self.n_features, dtype=torch.float64)

        self.weight_ih = nn.Parameter(torch.tensor([
            [0.6627, -0.4245], [ 0.5373,  0.2294]
        ], dtype=torch.float64, requires_grad=True))
        self.bias_ih = nn.Parameter(torch.tensor([0.4954, 0.6533], dtype=torch.float64, requires_grad=True))

        # Hidden
        self.weight_hh = nn.Parameter(torch.tensor([
            [-0.4015, -0.5385], [-0.1956, -0.6835]
        ], dtype=torch.float64, requires_grad=True))
        self.bias_hh = nn.Parameter(torch.tensor([-0.3565, -0.2904], dtype=torch.float64, requires_grad=True))

    def forward(self, point, hidden):
        """Takes in one at a time
        """
        x_cord, y_cord = point

        ix = x_cord * self.weight_ih[0][0] \
            + y_cord * self.weight_ih[0][1] \
            + self.bias_ih[0]

        iy = x_cord * self.weight_ih[1][0] \
            + y_cord * self.weight_ih[1][1] \
            + self.bias_ih[1]

        hx = hidden[0] * self.weight_hh[0][0] \
            + hidden[1] * self.weight_hh[0][1] \
            + self.bias_hh[0]

        hy = hidden[0] * self.weight_hh[1][0] \
            + hidden[1] * self.weight_hh[1][1] \
            + self.bias_hh[1]

        pretan_x = ix + hx
        pretan_y = iy + hy
        new_hidden = torch.stack([torch.tanh(pretan_x), torch.tanh(pretan_y)])

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

    def forward(self, four_points):
        ## Encoder
        hidden = [0., 0.]

        for i in range(2):      # 0, 1
            input_point = four_points[i]
            output_point, new_hidden = self.encoder(input_point, hidden)
            hidden = new_hidden
            # Note, we are just going to override input_point to use the actual points

        # here, hidden is the populated by the encoder

        ## Decoder
        y_hat = []

        input_point = four_points[1]     # x_1
        for i in range(2):
            # Applies RNN and Regression
            output_point, new_hidden = self.decoder(input_point, hidden)

            y_hat.append(output_point)
            hidden, input_point = new_hidden, output_point

        return y_hat




if __name__ == "__main__":

    torch.manual_seed(23)
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
    train_points, train_directions = data['points'], data['directions']
    test_points, test_directions = data['test_points'], data['test_directions']

    train_last2_points = [pts[2:] for pts in train_points]
    test_last2_points = [pts[2:] for pts in test_points]

    EPOCH = 20
    for epoch in range(EPOCH):
        model.train()

        y_hat = []
        for i, points in enumerate(train_points):
            y_hat.append(model(points))         # returns last 2 points

        # from chatgpt to convert list of points to tensor for loss fn
        y_hat_tensors = [torch.stack(tensors) for tensors in y_hat]  # This stacks tensors within each sublist
        y_hat_tensor = torch.cat(y_hat_tensors, dim=0)  # Concatenate along the first dimension
        train_last2_points_tensor = torch.Tensor(train_last2_points)  # This assumes train_last2_points is suitable for direct conversion
        train_last2_points_tensor = train_last2_points_tensor.view(-1, 2)  # Adjust the shape as necessary, here assuming each point is of dimension 2
        training_loss = loss(y_hat_tensor, train_last2_points_tensor)

        training_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            y_hat = []
            for i, points in enumerate(test_points):
                y_hat.append(model(points))

            # from chatgpt to convert list of points to tensor for loss fn
            y_hat_tensors = [torch.stack(tensors) for tensors in y_hat]  # This stacks tensors within each sublist
            y_hat_tensor = torch.cat(y_hat_tensors, dim=0)  # Concatenate along the first dimension
            test_last2_points_tensor = torch.Tensor(test_last2_points)  # This assumes test_last2_points is suitable for direct conversion
            test_last2_points_tensor = test_last2_points_tensor.view(-1, 2)  # Adjust the shape as necessary, here assuming each point is of dimension 2
            testing_loss = loss(y_hat_tensor, test_last2_points_tensor)

        print(f'Epoch:{epoch} training:{training_loss:,.3f} validation:{testing_loss:,.3f}')
