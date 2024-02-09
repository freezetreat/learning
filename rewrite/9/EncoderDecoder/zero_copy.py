

import torch.nn as nn
import torch
from torchviz import make_dot
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset


import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

import pickle
with open('random_data.pickle', 'rb') as inf:
    data = pickle.load(inf)

# points, directions = d['points'], d['directions']
# test_points, test_directions = d['test_points'], d['test_directions']


### Parsing
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-v', action='store_true')
parser.add_argument('-N', type=int, default=1)
parser.add_argument('-E', type=int, default=1)


# Parse the command line arguments.
args = parser.parse_args()

# Get the value of the "-v" argument.
VERBOSE = args.v
EPOCH = args.E
DATA_SIZE = args.N


class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.hidden = None
        self.basic_rnn = nn.GRU(self.n_features, self.hidden_dim, batch_first=True)

    def forward(self, X):
        rnn_out, self.hidden = self.basic_rnn(X)
        return rnn_out # N, L, F


class Decoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.hidden = None
        self.basic_rnn = nn.GRU(self.n_features, self.hidden_dim, batch_first=True)
        self.regression = nn.Linear(self.hidden_dim, self.n_features)

    def init_hidden(self, hidden_seq):
        # We only need the final hidden state
        hidden_final = hidden_seq[:, -1:] # N, 1, H
        # But we need to make it sequence-first
        self.hidden = hidden_final.permute(1, 0, 2) # 1, N, H

    def forward(self, X):
        # X is N, 1, F
        batch_first_output, self.hidden = self.basic_rnn(X, self.hidden)

        last_output = batch_first_output[:, -1:]
        out = self.regression(last_output)

        # N, 1, F
        return out.view(-1, 1, self.n_features)


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
        # splits the data in source and target sequences
        # the target seq will be empty in testing mode
        # N, L, F
        source_seq = X[:, :self.input_len, :]
        target_seq = X[:, self.input_len:, :]
        self.init_outputs(X.shape[0])

        # Encoder expected N, L, F
        hidden_seq = self.encoder(source_seq)
        # Output is N, L, H
        self.decoder.init_hidden(hidden_seq)

        # The last input of the encoder is also
        # the first input of the decoder
        dec_inputs = source_seq[:, -1:, :]

        # Generates as many outputs as the target length
        for i in range(self.target_len):
            # Output of decoder is N, 1, F
            out = self.decoder(dec_inputs)
            self.store_output(i, out)

            prob = self.teacher_forcing_prob
            # In evaluation/test the target sequence is
            # unknown, so we cannot use teacher forcing
            if not self.training:
                prob = 0

            # If it is teacher forcing
            if torch.rand(1) <= prob:
                # Takes the actual element
                dec_inputs = target_seq[:, i:i+1, :]
            else:
                # Otherwise uses the last predicted output
                dec_inputs = out

        return self.outputs


if __name__ == "__main__":

    from StepByStep import StepByStep

    torch.manual_seed(23)
    encoder = Encoder(n_features=2, hidden_dim=2)
    decoder = Decoder(n_features=2, hidden_dim=2)
    model = EncoderDecoder(encoder, decoder, input_len=2, target_len=2, teacher_forcing_prob=1.0)
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    sbs_seq = StepByStep(model, loss, optimizer)


    generator = torch.Generator()

    points, directions = data['points'], data['directions']
    full_train = torch.as_tensor(points).float()
    target_train = full_train[:, 2:]
    test_points, test_directions = data['test_points'], data['test_directions']
    full_test = torch.as_tensor(test_points).float()
    source_test = full_test[:, :2]
    target_test = full_test[:, 2:]


    train_data = TensorDataset(full_train, target_train)
    test_data = TensorDataset(source_test, target_test)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=False, generator=generator)
    test_loader = DataLoader(test_data, batch_size=16)


    sbs_seq.set_loaders(train_loader, test_loader)
    sbs_seq.train(10)

