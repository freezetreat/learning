

import torch.nn as nn
import torch
from torchviz import make_dot
from torch import optim

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

import pickle
with open('random_data.pickle', 'rb') as inf:
    d = pickle.load(inf)

points, directions = d['points'], d['directions']
test_points, test_directions = d['test_points'], d['test_directions']


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





### New Code

torch.manual_seed(21)
torch.set_default_dtype(torch.float64)

n_features = 2
hidden_dim = 2


# X is (1 item, 4 rows per item, 2 dimensions per row)
X = (
    torch.tensor([[-1, -1], [-2, -2], [3, 3], [4, 4]], dtype=torch.float64)
          .view(1, 4, 2)
    )
# print('X:\n', X)

# Showing how we use the first two points to get the next two points
source_seq = X[:, :2]       # tensor([[[-1., -1.], [-2., -2.]]])
target_seq = X[:, 2:]       # tensor([[[3., 3.], [4., 1.]]])


encoder_rnn = nn.GRU(n_features, hidden_dim, batch_first=True)
print("Encoder RNN state", encoder_rnn.state_dict())

decoder_rnn = nn.GRU(n_features, hidden_dim, batch_first=True)
print("Decoder RNN state", decoder_rnn.state_dict())

decoder_linear = nn.Linear(hidden_dim, n_features)
print("Decoder Linear state", decoder_linear.state_dict())


## Encoder

# output is the hidden states of the TWO points that you fed into it
output, encoder_hidden = encoder_rnn(source_seq)

# output: tensor([[[6.2459e-01, 4.6589e-05],
#         [9.3809e-01, 8.1336e-04]]], grad_fn=<TransposeBackward1>)
# encoder_hidden:
#       tensor([[[9.3809e-01, 8.1336e-04]]], grad_fn=<StackBackward0>)


## Decode

# Initial Hidden State will be encoder's final hidden state
decoder_output_1, decoder_hidden_1 = decoder_rnn(output, encoder_hidden)

output_1 = decoder_linear(decoder_output_1)

decoder_output_2, decoder_hidden_2 = decoder_rnn(output_1, decoder_hidden_1)

print(decoder_output_1)
print(decoder_output_2)



"""
Encoder
    Input: x0, x1
    Output: hidden state

Decoder
    Input: x1, hidden state
    Output: x2, x3

Thus a encoder should be judged on its ability to generate a good hidden state
while a decoder should be judged on its ability to generate the x2 and x3. However,
you can't really "judge" a hidden state.

Instead, we will first train the decoder, then train the encoder. But how do you get
the hidden state in the first place?
"""

