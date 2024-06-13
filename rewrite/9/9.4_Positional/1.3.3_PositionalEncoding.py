# Using a stupidly simple positional encoding module

import copy
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

from StepByStep import StepByStep


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

    def forward(self, query, target_mask=None):
        """
        target_mask is: tensor([ [True, False], [True, True] ])
        """
        Q = self.linear_query(query)
        res = torch.matmul(Q, self.K.transpose(-2, -1))
        res = res / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float64))
        if target_mask is not None:
            res = res.masked_fill(~target_mask, float('-inf'))
        res = F.softmax(res, dim=-1)
        return torch.matmul(res, self.V)


class Encoder(nn.Module):
    def __init__(self, n_features=0, hidden_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.attention = Attention(n_features=n_features, hidden_dim=hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, self.n_features),
        )

    def forward(self, X):
        self.attention.init(X)
        att = self.attention(X)
        return self.ffn(att)


class DecoderAttn(nn.Module):
    def __init__(self, n_features=0, hidden_dim=0):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.self_attention = Attention(n_features=n_features, hidden_dim=hidden_dim)
        self.cross_attention = Attention(n_features=n_features, hidden_dim=hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, self.n_features),
        )

    def init(self, encoder_states):
        self.cross_attention.init(encoder_states)

    def forward(self, X, target_mask=None):
        self.self_attention.init(X)
        out = self.self_attention(X, target_mask)
        out = self.cross_attention(out)
        return self.ffn(out)


class StupidPositionalEncoding(nn.Module):

    def forward(self, X):
        """
        Just adds [1, 1], [2, 2], [3, 3], ...
        """
        N, L, F = X.size()
        # positional_encoding = torch.arange(1, N + 1).view(N, 1, 1).expand(N, L, F)
        positional_encoding = torch.arange(1, N + 1).float() * 0.0001
        positional_encoding = positional_encoding.view(N, 1, 1).expand(N, L, F)
        return X + positional_encoding


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        angular_speed = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * angular_speed) # even dimensions
        pe[:, 1::2] = torch.cos(position * angular_speed) # odd dimensions
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        scaled_x = x * np.sqrt(self.d_model)
        encoded = scaled_x + self.pe[:, :x.size(1), :]
        return encoded


class OverallModel(nn.Module):
    def __init__(self, encoder, decoder, n_features=2):
        super().__init__()
        self.encoder = encoder
        self.encoder_pe = PositionalEncoding(2, 2)

        self.decoder = decoder
        self.decoder_pe = PositionalEncoding(2, 2)

        self.n_features = n_features

    def forward(self, X):
        """
        Using parallel processing now.
        Instead of point by point to our decoder, we will feed the shifted target sequence
        This means we are using 100% teacher forcing. Also, X now has 4 points instead of 2
        """
        source_seq = X[:, :2, :]

        positioned = self.encoder_pe(source_seq)
        encoder_output = self.encoder(positioned)
        self.decoder.init(encoder_output)

        target_mask = torch.tensor([
            [True, False],          # first row can't peak
            [True, True]            # second row is ok
        ])

        if self.training:
            # points: [0, 1, 2, 3], shifted: [1, 2]
            shifted_target_seq = X[:, 1:3, :]

            positioned = self.decoder_pe(shifted_target_seq)
            return self.decoder(positioned, target_mask=target_mask)
        else:
            # generates a sequence one point at a time
            inputs = source_seq[:, -1:]     # last point from source seq
            outputs = torch.zeros(batch_size, 2, self.n_features, dtype=torch.float64)

            for i in range(2):
                # No need trig mask anymore, everything is visible
                positioned = self.decoder_pe(inputs)
                out = self.decoder(positioned)
                outputs[:, i:i+1, :] = out[:, -1:, :]
                inputs = torch.cat([inputs, out[:, -1:, :]], dim=1)

            return outputs



if __name__ == "__main__":
    torch.manual_seed(23)
    torch.set_default_dtype(torch.float64)

    encoder = Encoder(n_features=2, hidden_dim=2)
    decoder_attn = DecoderAttn(n_features=2, hidden_dim=2)
    model = OverallModel(encoder, decoder_attn)
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


