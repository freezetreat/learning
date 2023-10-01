# instead of going point by point, use the batch approach
# (you need to get used to it anyway eventually)

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
parser.add_argument('-N', type=int, default=1000000)
parser.add_argument('-E', type=int, default=100)


# Parse the command line arguments.
args = parser.parse_args()

# Get the value of the "-v" argument.
VERBOSE = args.v
EPOCH = args.E
DATA_SIZE = args.N


### RNN
torch.manual_seed(19)
torch.set_default_dtype(torch.float64)

n_features = 2
hidden_dim = 2

# NOTE!! This is not rnn_cell but RNN!!!
rnn_whole = nn.GRU(n_features, hidden_dim)
rnn_state = rnn_whole.state_dict()
print("RNN coefficients\n", rnn_state)

## Don't touch the classifier, our focus is on RNN rather than classifier
classifier = nn.Linear(hidden_dim, 1)
classifier.weight.data = torch.tensor([[-0.2732, -0.1587]], dtype=torch.float64)
classifier.bias.data = torch.tensor([0.5806], dtype=torch.float64)
# ALWAYS CHECK DEFAULT WEIGHTS. THEY MIGHT CHANGE AFTER YOU CHANGE DATA TYPES
print('classifier coefficients\n', classifier.state_dict())

# loss = nn.BCELoss()           # expects number from 0 to 1
loss = nn.BCEWithLogitsLoss()   # just sigmod with BCELoss

optimizer = optim.Adam(list(rnn_whole.parameters()) + list(classifier.parameters()), lr=0.01)


# To use the batch approach, points and directions needs to be tensors
points = torch.tensor(points, dtype=torch.float64)
directions = torch.tensor(directions, dtype=torch.float64)

if VERBOSE:
    print(points.shape)         # (3 points, 4 rows/point, 2 features per row )
    print(directions.shape)     # (3 points)

# NOTE: !! You can't just feed (3, 4, 2) into the RNN!!
# It will think you have 4 points, each with 3 rows and 2 features!!
#
# For the GRU (and most other RNNs in PyTorch), the expected input shape is (seq_len, batch_size, input_size). The output shapes are as follows:
#
# output of shape (seq_len, batch_size, num_directions * hidden_size)
# hidden of shape (num_layers * num_directions, batch_size, hidden_size)
# In your case:
#
# seq_len is 4 (since each point has 4 rows)
# batch_size is 3 (since you input 3 points)
# input_size is 2 (as each row has 2 features)
# hidden_size is 2 (as defined by hidden_dim)
# Therefore, you should expect output to have the shape (4, 3, 2) and hidden to have the shape (1, 3, 2) (since you're using a single-layer unidirectional GRU).
#
# Change from (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
points = points.transpose(0, 1)


points, directions = points[:DATA_SIZE], directions[:DATA_SIZE]

for epoch in range(EPOCH):

    ###################### Y_hat ######################
    classifier_outputs = []

    # Now instead of feeding point by point, we will by using it as a batch
    output, hidden = rnn_whole(points)

    # we want to classify on the HIDDEN

    # We will feed the last "out" to the classifier
    if VERBOSE:
        print('What are we feeding into the classifier?', hidden)
    temp = classifier(hidden)
    if VERBOSE:
        print('What comes out from the classifier?', temp)

    classifier_outputs.append(temp)

    ###################### end of Y_hat ######################

    # Convert [ tensor, .. ] to tensor([float, ...])
    # print("Classifier (Before)", classifier_outputs)
    classifier_outputs_tensor = torch.cat(classifier_outputs).view(-1).to(torch.float64)
    if VERBOSE:
        print("Classifier (After)", classifier_outputs_tensor)

    # Convert directions numpy array to a PyTorch tensor
    # print("Directions (Before)", directions)
    directions_tensor = torch.tensor(directions, dtype=torch.float64)
    # print("Directions (After)", directions_tensor)

    # Now we need to compute loss
    training_loss = loss(classifier_outputs_tensor, directions_tensor)
    print(f"Epoch:{epoch}, training_loss:{training_loss.data}")

    training_loss.backward()
    optimizer.step()
    optimizer.zero_grad()


