

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

# Parse the command line arguments.
args = parser.parse_args()

# Get the value of the "-v" argument.
VERBOSE = args.v



### RNN
torch.manual_seed(19)
torch.set_default_dtype(torch.float64)

n_features = 2
hidden_dim = 2

rnn_cell = nn.GRUCell(input_size=n_features, hidden_size=hidden_dim)
rnn_state = rnn_cell.state_dict()
print("RNN coefficients\n", rnn_state)

## Don't touch the classifier, our focus is on RNN rather than classifier
classifier = nn.Linear(hidden_dim, 1)
classifier.weight.data = torch.tensor([[-0.2732, -0.1587]], dtype=torch.float64)
classifier.bias.data = torch.tensor([0.5806], dtype=torch.float64)
# ALWAYS CHECK DEFAULT WEIGHTS. THEY MIGHT CHANGE AFTER YOU CHANGE DATA TYPES
print('classifier coefficients\n', classifier.state_dict())

EPOCH = 100
points, directions = points[:1], directions[:1]

# loss = nn.BCELoss()           # expects number from 0 to 1
loss = nn.BCEWithLogitsLoss()   # just sigmod with BCELoss

optimizer = optim.Adam(list(rnn_cell.parameters()) + list(classifier.parameters()), lr=0.01)

for epoch in range(EPOCH):

    ###################### Y_hat ######################
    classifier_outputs = []

    for i, point in enumerate(points):
        hidden = torch.zeros(1, hidden_dim)
        if VERBOSE:
            print('initial hidden', hidden)

        X = torch.as_tensor(point)
        if VERBOSE:
            print("Input:", X.data)

        out = None
        for i in range(X.shape[0]):
            out = rnn_cell(X[i:i+1], hidden)
            hidden = out
            if VERBOSE:
                print(f"Step {i}: hidden:{hidden.data}")

        # We will feed the last "out" to the classifier
        if VERBOSE:
            print('What are we feeding into the classifier?', out)
        temp = classifier(out)
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


