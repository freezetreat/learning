# Batching instead now. Might look very similar to 0

import copy
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

from StepByStep import StepByStep


torch.set_default_dtype(torch.float64)
torch.manual_seed(23)


class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.hidden = None
        self.basic_rnn = nn.RNN(self.n_features, self.hidden_dim, dtype=torch.float64, batch_first=True)
        with torch.no_grad():
            # Encoder RNN weights
            self.basic_rnn.weight_ih_l0 = nn.Parameter(torch.tensor([[-0.1046, -0.1705],
                                                                    [0.2285, -0.2168]]))
            self.basic_rnn.weight_hh_l0 = nn.Parameter(torch.tensor([[0.2841, -0.1657],
                                                                    [-0.2813, 0.3142]]))
            self.basic_rnn.bias_ih_l0 = nn.Parameter(torch.tensor([0.3835, 0.1879]))
            self.basic_rnn.bias_hh_l0 = nn.Parameter(torch.tensor([-0.6391, -0.0030]))

    def forward(self, X):
        rnn_out, self.hidden = self.basic_rnn(X)
        return rnn_out # N, L, F


class DecoderAttn(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.hidden = None
        self.basic_rnn = nn.RNN(self.n_features, self.hidden_dim, batch_first=True, dtype=torch.float64)
        # DecoderAttn RNN weights
        with torch.no_grad():
            self.basic_rnn.weight_ih_l0 = nn.Parameter(torch.tensor([[-0.1802, -0.3691],
                                                                    [-0.0514, 0.4502]]))
            self.basic_rnn.weight_hh_l0 = nn.Parameter(torch.tensor([[0.3566, -0.3189],
                                                                    [0.1933, 0.2683]]))
            self.basic_rnn.bias_ih_l0 = nn.Parameter(torch.tensor([-0.1258, -0.1091]))
            self.basic_rnn.bias_hh_l0 = nn.Parameter(torch.tensor([-0.3417, -0.5897]))

        # Attention is a submodule of decoder
        self.attn = Attention(self.hidden_dim)

        # Note the 2x here. We need to account for the concatenation of the context
        self.regression = nn.Linear(2 * self.hidden_dim, self.n_features, dtype=torch.float64)
        with torch.no_grad():
            self.regression.weight = nn.Parameter(torch.tensor([[-0.3285, -0.3166, -0.4496, 0.2151],
                                                                [-0.1001, 0.0352, -0.2030, -0.1448]]))
            self.regression.bias = nn.Parameter(torch.tensor([-0.0187, 0.2951]))

    def init_hidden(self, hidden_seq):
        """Storing the hidden output from the encoder. Note that the hidden output
        is already computed so it won't change. What will change is the similarity
        score that gets applied to each of them for each step of the decoder.
        """
        # the output of the encoder is N, L, H
        # and init_keys expects batch-first as well
        self.attn.init_keys(hidden_seq)
        hidden_final = hidden_seq[:, -1:]
        self.hidden = hidden_final.permute(1, 0, 2)   # L, N, H

    def forward(self, X, mask=None):
        # X is N, 1, F
        # batch size of N, 1 point, f features. Takes in 1 point at a time.
        batch_first_output, self.hidden = self.basic_rnn(X, self.hidden)

        # What changed here is that instead of feeding the output of the RNN (not
        # hidden) directly to the regression layer, we feed it to this attention
        # mechanism. Note that for our original RNN, the output is the same as the
        # hidden.
        query = batch_first_output[:, -1:]
        # Attention
        context = self.attn(query, mask=mask)
        concatenated = torch.cat([context, query], axis=-1)
        out = self.regression(concatenated)

        # N, 1, F
        return out.view(-1, 1, self.n_features)


### Implementation


class Attention(nn.Module):
    def __init__(self, hidden_dim, input_dim=None, proj_values=False):
        super().__init__()
        self.d_k = hidden_dim
        self.input_dim = hidden_dim if input_dim is None else input_dim
        self.proj_values = proj_values

        # Initialized keys and values will be set in init_keys
        self.keys = None
        self.values = None
        self.alphas = None

        # Affine transformations for Q, K, and V
        self.linear_query = nn.Linear(self.input_dim, hidden_dim, dtype=torch.float64)
        self.linear_key = nn.Linear(self.input_dim, hidden_dim, dtype=torch.float64)
        self.linear_value = nn.Linear(self.input_dim, hidden_dim, dtype=torch.float64)

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

    def init_keys(self, keys):
        """Given the hidden states from the encoder for all x0, x1, compute the keys
        and store it so that for different queries we can just reuse them

        when batch=3
        keys tensor([[[-0.5061,  0.1872],
                     [-0.3318,  0.7027]],

                    [[-0.4729,  0.2452],
                    [-0.3328,  0.6934]],

                    [[-0.1814,  0.5600],
                    [-0.5750,  0.3777]]], grad_fn=<TransposeBackward1>)
        """
        # You want each key to be mapped to self.linear_key
        # somethiing along the lines of: self.keys = torch.matmul(keys, self.linear_key.weights) + self.linear_key.bias
        self.keys = keys
        self.proj_keys = self.linear_key(keys)

        # we are not doing anything to the values
        self.values = keys

    def forward(self, query_batched, mask=None):
        # Implement softmax (QKT)/sqrt(dk)

        # To fix this issue, you should specify the dimensions along which you want
        # to transpose self.proj_keys. In the context of computing the dot product
        # between query, key, and value matrices in the attention mechanism, you
        # typically want to transpose the keys along the last two dimensions. So, you
        # can modify that line in your forward method to transpose self.proj_keys
        # along dimensions 1 and 2.
        qkt = torch.bmm(self.linear_query(query_batched), self.proj_keys.transpose(1, 2))
        pre_softmax = qkt / np.sqrt(self.d_k)

        # In PyTorch, when applying operations like softmax along a particular dimension of a tensor, the dim parameter specifies the dimension along which the operation is performed. In this case, dim=-1 indicates that the softmax operation is applied along the last dimension of the pre_softmax tensor.
        # Here's a breakdown of why dim=-1 might be used in this context:

        # The pre_softmax tensor likely has shape [batch_size, num_queries, num_keys].
        # Applying softmax along the last dimension (dim=-1) means that softmax will
        # be computed for each row of the tensor independently. This operation
        # ensures that the values in each row sum up to 1, representing a probability
        # distribution over the keys for each query.
        self.alphas = F.softmax(pre_softmax, dim=-1)

        return torch.bmm(self.alphas, self.values)




### End of implementation here


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
        """
        > Question: how come the encoder part doesn't need to loop
        whereas the decode part needs to?

        The encoder processes the entire input sequence at once to capture its
        overall context into a set of hidden states.  The decoder generates the
        output sequence one step at a time, using the context provided by the encoder
        and the dependencies between the elements of the output sequence.
        """
        # splits the data in source and target sequences
        # the target seq will be empty in testing mode
        # N, L, F
        source_seq = X[:, :self.input_len, :]
        target_seq = X[:, self.input_len:, :]
        self.init_outputs(X.shape[0])

        # Encoder expected N, L, F
        hidden_seq = self.encoder(source_seq)
        # Output is N, L, H

        """
        we are assigning the hidden output from the RNN for all input sequence
        to the decoderu
        """
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


class EncoderDecoderAttn(EncoderDecoder):
    def __init__(self, encoder, decoder, input_len, target_len, teacher_forcing_prob=0.5):
        super().__init__(encoder, decoder, input_len, target_len, teacher_forcing_prob)
        self.alphas = None

    def init_outputs(self, batch_size):
        device = next(self.parameters()).device
        # N, L (target), F
        self.outputs = torch.zeros(batch_size,
                              self.target_len,
                              self.encoder.n_features).to(device)
        # N, L (target), L (source)
        self.alphas = torch.zeros(batch_size,
                                  self.target_len,
                                  self.input_len).to(device)

    def store_output(self, i, out):
        # Stores the output
        self.outputs[:, i:i+1, :] = out
        self.alphas[:, i:i+1, :] = self.decoder.attn.alphas



if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)

    encoder = Encoder(n_features=2, hidden_dim=2)
    decoder_attn = DecoderAttn(n_features=2, hidden_dim=2)
    model = EncoderDecoderAttn(encoder, decoder_attn, input_len=2, target_len=2, teacher_forcing_prob=0)
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    import pickle
    with open('random_data.pickle', 'rb') as inf:
        data = pickle.load(inf)

    points, directions = data['points'], data['directions']
    full_train = torch.as_tensor(points).double()
    target_train = full_train[:, 2:]

    test_points, test_directions = data['test_points'], data['test_directions']
    full_test = torch.as_tensor(test_points).double()
    source_test = full_test[:, :2]
    target_test = full_test[:, 2:]

    train_data = TensorDataset(full_train, target_train)
    test_data = TensorDataset(source_test, target_test)

    generator = torch.Generator()
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, generator=generator)
    test_loader = DataLoader(test_data, batch_size=1)

    sbs_seq = StepByStep(model, loss, optimizer)
    sbs_seq.set_loaders(train_loader, test_loader)
    sbs_seq.train(20)


