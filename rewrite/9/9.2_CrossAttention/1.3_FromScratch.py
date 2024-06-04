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

        with torch.no_grad():
            self.rnn.weight_ih_l0 = nn.Parameter(torch.tensor([[-0.1046, -0.1705],
                                                               [0.2285, -0.2168]]))
            self.rnn.weight_hh_l0 = nn.Parameter(torch.tensor([[0.2841, -0.1657],
                                                               [-0.2813, 0.3142]]))
            self.rnn.bias_ih_l0 = nn.Parameter(torch.tensor([0.3835, 0.1879]))
            self.rnn.bias_hh_l0 = nn.Parameter(torch.tensor([-0.6391, -0.0030]))

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

        with torch.no_grad():
            self.linear_query.weight = nn.Parameter(torch.tensor([[0.0798, 0.4151],
                                                                 [-0.0994, 0.1561]]))
            self.linear_query.bias = nn.Parameter(torch.tensor([-0.2548, 0.3911]))
            self.linear_key.weight = nn.Parameter(torch.tensor([[-0.3068, -0.4800],
                                                                [-0.4578, -0.1488]]))
            self.linear_key.bias = nn.Parameter(torch.tensor([0.3407, 0.4099]))
            self.linear_value.weight = nn.Parameter(torch.tensor([[-0.2710, -0.6198],
                                                                 [0.4265, -0.3488]]))
            self.linear_value.bias = nn.Parameter(torch.tensor([-0.3975, -0.1983]))

    def init(self, encoder_states):
        self.K = self.linear_key(encoder_states)
        # even though V is supposed to go right through, we still need to map it to
        # the right dimensions
        self.V = self.linear_value(encoder_states)

    def forward(self, query):
        Q = self.linear_query(query)
        res = torch.matmul(Q, self.K.transpose(-2, -1))
        res = res / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float64))
        res = F.softmax(res, dim=-1)
        return torch.matmul(res, self.V)


class DecoderAttn(nn.Module):
    def __init__(self, n_features=0, hidden_dim=0):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.attention = Attention(n_features=n_features, hidden_dim=hidden_dim)
        self.rnn = nn.RNN(self.n_features, self.hidden_dim, dtype=torch.float64, batch_first=True)
        self.linear = nn.Linear(2 * self.n_features, self.n_features, dtype=torch.float64)

        with torch.no_grad():
            self.rnn.weight_ih_l0 = nn.Parameter(torch.tensor([[-0.1802, -0.3691],
                                                               [-0.0514, 0.4502]]))
            self.rnn.weight_hh_l0 = nn.Parameter(torch.tensor([[0.3566, -0.3189],
                                                               [0.1933, 0.2683]]))
            self.rnn.bias_ih_l0 = nn.Parameter(torch.tensor([-0.1258, -0.1091]))
            self.rnn.bias_hh_l0 = nn.Parameter(torch.tensor([-0.3417, -0.5897]))

        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.tensor([[-0.3285, -0.3166, -0.4496, 0.2151],
                                                            [-0.1001, 0.0352, -0.2030, -0.1448]]))
            self.linear.bias = nn.Parameter(torch.tensor([-0.0187, 0.2951]))

    def forward(self, query, hidden):
        output, hidden = self.rnn(query, hidden)
        query = output[:, -1:]

        context = self.attention(query)
        concatenated = torch.cat([context, query], axis=-1)
        out = self.linear(concatenated)
        return out.view(-1, 1, self.n_features), hidden

    def init(self, encoder_output):
        self.attention.init(encoder_output)


class EncoderDecoderAttn(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X):
        self.outputs = torch.zeros(
            X.shape[0],
            2,      # 2 points output
            self.encoder.n_features)

        output, hidden = self.encoder(X)
        self.decoder.init(output)

        dec_inputs = output[:, -1:, :]
        for i in range(2):
            dec_inputs, hidden = self.decoder(dec_inputs, hidden)
            self.outputs[:, i:i+1, :] = dec_inputs

        return self.outputs


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
    source_train = full_train[:, :2]
    target_train = full_train[:, 2:]

    test_points, test_directions = data['test_points'], data['test_directions']
    full_test = torch.as_tensor(test_points).double()
    source_test = full_test[:, :2]
    target_test = full_test[:, 2:]

    train_data = TensorDataset(source_train, target_train)
    test_data = TensorDataset(source_test, target_test)

    generator = torch.Generator()
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, generator=generator)
    test_loader = DataLoader(test_data, batch_size=16)

    sbs_seq = StepByStep(model, loss, optimizer)
    sbs_seq.set_loaders(train_loader, test_loader)
    sbs_seq.train(100)


