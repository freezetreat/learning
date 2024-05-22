# changing this to single attention to make things more legible.
# Note that performance decreased slightly

import copy
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

# from data_generation.square_sequences import generate_sequences
from StepByStep import StepByStep


torch.set_default_dtype(torch.float64)
torch.manual_seed(23)


# Refer to this: https://camo.githubusercontent.com/58d0a0d0dfb1a4339d7680614eb4a9ee6afd91709bee2bc177e4b79ab48c6d24/68747470733a2f2f6769746875622e636f6d2f6476676f646f792f5079546f726368537465704279537465702f626c6f622f6d61737465722f696d616765732f656e636465635f73656c665f73696d706c69666965642e706e673f7261773d31



class Attention(nn.Module):
    """
    In encoder:
    input: [ [x00, x01], [x10, x11] ]
    output: [ [h00, h01], [h10, h11] ]

    In decoder:
    input: [ [h00, h01], [h10, h11] ]
    output: [ [c20, c21] ]
    """
    def __init__(self, hidden_dim, input_dim=None, proj_values=False):
        super().__init__()
        self.d_k = hidden_dim
        self.input_dim = hidden_dim if input_dim is None else input_dim
        self.proj_values = proj_values
        # Affine transformations for Q, K, and V
        self.linear_query = nn.Linear(self.input_dim, hidden_dim)
        self.linear_key = nn.Linear(self.input_dim, hidden_dim)
        self.linear_value = nn.Linear(self.input_dim, hidden_dim)

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


        self.alphas = None

    def init_keys(self, keys):
        self.keys = keys
        self.proj_keys = self.linear_key(self.keys)
        self.values = self.linear_value(self.keys) \
                      if self.proj_values else self.keys

    def score_function(self, query):
        proj_query = self.linear_query(query)

        # Q.K_T / sqrt (D_k)
        # scaled dot product
        # N, 1, H x N, H, L -> N, 1, L
        dot_products = torch.bmm(proj_query, self.proj_keys.permute(0, 2, 1))

        scores =  dot_products / np.sqrt(self.d_k)
        return scores

    def forward(self, query, mask=None):
        # Query is batch-first N, 1, H
        scores = self.score_function(query) # N, 1, L
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        alphas = F.softmax(scores, dim=-1) # N, 1, L
        self.alphas = alphas.detach()
        # alphas tensor([[[0.4311, 0.5689],
        #                 [0.5861, 0.4139]]])

        # N, 1, L x N, L, H -> N, 1, H
        # 1xL x LxH -> 1xH
        context = torch.bmm(alphas, self.values)
        return context


class EncoderSelfAttn(nn.Module):
    """
    The goal of the encoder is to take
    [
        [x00, x01], [x10, x11]
    ]

    and convert it to:
    [
        [h00, h01], [h10, h11]
    ]

    (red to blue in the diagram)
    """

    def __init__(self, n_heads, d_model, ff_units, n_features=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.ff_units = ff_units
        self.n_features = n_features
        self.self_attn = Attention(d_model, input_dim=n_features)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.ReLU(),
            nn.Linear(ff_units, d_model),
        )

    def forward(self, query, mask=None):
        self.self_attn.init_keys(query)
        att = self.self_attn(query, mask)
        out = self.ffn(att)
        return out


class DecoderSelfAttn(nn.Module):
    """
    The goal of the decoder is to take:
    - last point: [x10, x11]
    - hidden states from encoder: [ [h00, h01], [h10, h11] ]
    to generate [x20, x21]

    Similarly, to generate [x30, x31], we need:
    - last point: [x20, x21]
    - hidden states from encoder: [ [h00, h01], [h10, h11] ]
    """
    def __init__(self, n_heads, d_model, ff_units, n_features=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.ff_units = ff_units
        self.n_features = d_model if n_features is None else n_features
        self.self_attn_heads = Attention(d_model, input_dim=self.n_features)
        self.cross_attn_heads = Attention(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.ReLU(),
            nn.Linear(ff_units, self.n_features),
        )

    def init_keys(self, states):
        self.cross_attn_heads.init_keys(states)

    def forward(self, query, source_mask=None, target_mask=None):
        """
        query tensor([[[ 0.8055, -0.9169],
                       [-0.8251, -0.9499]]])
        target_mask tensor([[[ True, False],
                             [ True,  True]]])
        """
        # Source Mask: Applied to the outputs of the encoder to control which
        # positions in the input sequence can be attended to during cross-attention
        # in the decoder. It's useful for masking out padding tokens or otherwise
        # specifying which parts of the input are relevant.
        #
        # Target Mask: Applied during self-attention within the decoder to prevent
        # positions from attending to future positions in the output sequence,
        # ensuring the model cannot cheat by using future information to predict the
        # current output. It enforces the autoregressive property of the model during
        # training.

        # Here, we pass in [x10, x11] and [x20, x21]

        self.self_attn_heads.init_keys(query)
        att1 = self.self_attn_heads(query, target_mask)
        att2 = self.cross_attn_heads(att1, source_mask)
        out = self.ffn(att2)
        return out


class EncoderDecoderSelfAttn(nn.Module):
    def __init__(self, encoder, decoder, input_len, target_len):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_len = input_len
        self.target_len = target_len
        self.trg_masks = self.subsequent_mask(self.target_len)

    @staticmethod
    def subsequent_mask(size):
        attn_shape = (1, size, size)
        subsequent_mask = (1 - torch.triu(torch.ones(attn_shape), diagonal=1))
        return subsequent_mask

    def encode(self, source_seq, source_mask):
        # Encodes the source sequence and uses the result
        # to initialize the decoder

        # source_seq=tensor([[[ 1.0349,  0.9661],
        #                     [ 0.8055, -0.9169]]])
        # source_mask=None
        encoder_states = self.encoder(source_seq, source_mask)

        # Here, the encoder takes the source_seq, applies attention, and returns the
        # output of the feed forward network (which is the hidden state of the input).
        # those hidden states will be used for cross attention in the decoder

        self.decoder.init_keys(encoder_states)

    def decode(self, shifted_target_seq, source_mask=None, target_mask=None):
        # Decodes/generates a sequence using the shifted (masked)
        # target sequence - used in TRAIN mode

        # print(source_mask)        => None
        # print(target_mask)        => tensor([[[ True, False],
        #                                       [ True,  True]]])
        outputs = self.decoder(shifted_target_seq,
                               source_mask=source_mask,
                               target_mask=target_mask)

        return outputs

    def predict(self, source_seq, source_mask):
        # Decodes/generates a sequence using one input
        # at a time - used in EVAL mode
        # source_mask is not used

        # inputs is the last point of all rows in the batch
        inputs = source_seq[:, -1:]
        # inputs = tensor([[[0.9756, 0.9764]]])

        """
        i=0
        trg_masks=tensor([[[True]]])
        out=tensor([[[ 0.9756,  0.9764],
                     [-0.0934, -0.9373]]])

        i=1
        trg_masks=tensor([[[ True, False],
                           [ True,  True]]])
        out=tensor([[[ 0.9756,  0.9764],
                     [-0.0934, -0.9373],
                     [-0.0911, -0.9380]]])
        """

        for i in range(self.target_len):
            # Then we generate the trg_mask and feed it into the decoder
            out = self.decode(inputs, source_mask, self.trg_masks[:, :i+1, :i+1])

            # we concatenate inputs with the last point generated
            out = torch.cat([inputs, out[:, -1:, :]], dim=-2)

            # detach it
            inputs = out.detach()

        """
        i=0
        outputs=tensor([[[-0.0934, -0.9373]]])
        i=1
        outputs=tensor([[[-0.0934, -0.9373],
                         [-0.0911, -0.9380]]])
        """
        # And we take the second point onwards, since the first is the last input
        outputs = inputs[:, 1:, :]
        return outputs

    def forward(self, X, source_mask=None):
        # X contains all 4 points, it is a tensor of (N, 4 points, 2 dimension per point)

        # X=tensor([[[ 1.0349,  0.9661],
        #            [ 0.8055, -0.9169],
        #            [-0.8251, -0.9499],
        #            [-0.8670,  0.9342]]])

        # Sends the mask to the same device as the inputs
        self.trg_masks = self.trg_masks.type_as(X).bool()
        # trg_masks tensor([[[ True, False],
        #                    [ True,  True]]])

        # Slices the input to get source sequence
        source_seq = X[:, :self.input_len, :]
        # source_seq tensor([[[ 1.0349,  0.9661],
        #                     [ 0.8055, -0.9169]]])

        # Encodes source sequence AND initializes decoder
        # This step just performs self attention on the encoder and passes the hidden
        # states to the decoder (which is not activated yet)
        self.encode(source_seq, source_mask)

        if self.training:
            # WITH TEACHER FORCING

            # Slices the input to get the shifted target seq
            """
            X tensor([[[ 1.0349,  0.9661],
                       [ 0.8055, -0.9169],
                       [-0.8251, -0.9499],
                       [-0.8670,  0.9342]]])
            shifted tensor([[[ 0.8055, -0.9169],
                             [-0.8251, -0.9499]]])
            """
            shifted_target_seq = X[:, self.input_len-1:-1, :]

            # Decodes using the trg_mask to prevent cheating
            outputs = self.decode(shifted_target_seq, source_mask, self.trg_masks)
            # outputs tensor([[[-0.1157, -0.0468],
            #                  [-0.1157, -0.0468]]], grad_fn=<ViewBackward0>)
            # well they are same because they are uninitiailized

        else:
            # WITHOUT teacher forcing

            # Decodes using its own predictions
            outputs = self.predict(source_seq, source_mask)

        # print(f'X={X}, outputs={outputs}')
        return outputs


if __name__ == "__main__":
    encself = EncoderSelfAttn(n_heads=3, d_model=2, ff_units=10, n_features=2)
    decself = DecoderSelfAttn(n_heads=3, d_model=2, ff_units=10, n_features=2)
    model = EncoderDecoderSelfAttn(encself, decself, input_len=2, target_len=2)
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
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, generator=generator)
    test_loader = DataLoader(test_data, batch_size=1)

    sbs_seq_selfattn = StepByStep(model, loss, optimizer)
    sbs_seq_selfattn.set_loaders(train_loader, test_loader)
    sbs_seq_selfattn.train(20)