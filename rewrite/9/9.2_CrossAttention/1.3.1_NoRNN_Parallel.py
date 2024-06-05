# Remove the RNN layer from the decoder.
# Use the attention mechanism to compute the context vector.
# Concatenate the context vector with the query and pass it through a feedforward network to get the final output.

import copy
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

from StepByStep import StepByStep


class Encoder(nn.Module):
    def __init__(self, n_features=0, hidden_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.rnn = nn.RNN(self.n_features, self.hidden_dim, dtype=torch.float64, batch_first=True)

    def forward(self, query):
        output, hidden = self.rnn(query)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, n_features=0, hidden_dim=0):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        self.linear_key = nn.Linear(self.n_features, hidden_dim, dtype=torch.float64)
        self.linear_query = nn.Linear(self.n_features, hidden_dim, dtype=torch.float64)
        self.linear_value = nn.Linear(self.n_features, hidden_dim, dtype=torch.float64)

        self.K = None
        self.V = None

    def init(self, encoder_states):
        # Note that encoder_states is now all the states of the encoder

        self.K = self.linear_key(encoder_states)
        # even though V is supposed to go right through, we still need to map it to
        # the right dimensions
        self.V = self.linear_value(encoder_states)

    def forward(self, query, target_mask):
        """
        target_mask is: tensor([ [True, False], [True, True] ])
        """
        Q = self.linear_query(query)
        res = torch.matmul(Q, self.K.transpose(-2, -1))
        res = res / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float64))
        res = res.masked_fill(~target_mask, float('-inf'))
        res = F.softmax(res, dim=-1)
        return torch.matmul(res, self.V)


class DecoderAttn(nn.Module):
    def __init__(self, n_features=0, hidden_dim=0):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.attention = Attention(n_features=n_features, hidden_dim=hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_features),
        )

    def init(self, encoder_states):
        self.attention.init(encoder_states)

    def forward(self, X, target_mask=None):
        context = self.attention(X, target_mask)
        # No more concatenating, directly feed context to linear
        out = self.ffn(context)
        return out


class EncoderDecoderAttn(nn.Module):
    def __init__(self, encoder, decoder, n_features=2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_features = n_features

    def forward(self, X):
        """
        Using parallel processing now.
        Instead of point by point to our decoder, we will feed the shifted target sequence
        This means we are using 100% teacher forcing. Also, X now has 4 points instead of 2
        """
        source_seq = X[:, :2, :]
        encoder_states, last_hidden = self.encoder(source_seq)
        self.decoder.init(encoder_states)

        target_mask = torch.tensor([
            [True, False],          # first row can't peak
            [True, True]            # second row is ok
        ])

        if self.training:
            # points: [0, 1, 2, 3], shifted: [1, 2]
            shifted_target_seq = X[:, 1:3, :]

            return self.decoder(shifted_target_seq, target_mask=target_mask)
        else:
            # generates a sequence one point at a time
            inputs = source_seq[:, -1:]     # last point from source seq
            outputs = torch.zeros(batch_size, 2, self.n_features, dtype=torch.float64)

            for i in range(2):
                out = self.decoder(inputs, target_mask)      # Fix the target_mask indexing
                outputs[:, i:i+1, :] = out[:, -1:, :]
                inputs = torch.cat([inputs, out[:, -1:, :]], dim=1)

            return outputs



if __name__ == "__main__":
    torch.manual_seed(23)
    torch.set_default_dtype(torch.float64)

    encoder = Encoder(n_features=2, hidden_dim=2)
    decoder_attn = DecoderAttn(n_features=2, hidden_dim=2)
    model = EncoderDecoderAttn(encoder, decoder_attn)
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    import pickle
    with open('random_data.pickle', 'rb') as inf:
        data = pickle.load(inf)

    points, directions = data['points'], data['directions']
    full_train = torch.as_tensor(points).double()
    source_train = full_train[:, :4]        # we need all 4 points now
    target_train = full_train[:, 2:]

    test_points, test_directions = data['test_points'], data['test_directions']
    full_test = torch.as_tensor(test_points).double()
    source_test = full_test[:, :2]
    target_test = full_test[:, 2:]

    train_data = TensorDataset(source_train, target_train)
    test_data = TensorDataset(source_test, target_test)

    batch_size = 16

    generator = torch.Generator()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    sbs_seq = StepByStep(model, loss, optimizer)
    sbs_seq.set_loaders(train_loader, test_loader)
    sbs_seq.train(100)


