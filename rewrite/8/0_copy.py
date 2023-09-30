

import torch.nn as nn
import torch
from torchviz import make_dot

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

import pickle
with open('random_data.pickle', 'rb') as inf:
    d = pickle.load(inf)

points, directions = d['points'], d['directions']
test_points, test_directions = d['test_points'], d['test_directions']



### RNN

hidden_state = torch.zeros(2)

n_features = 2
hidden_dim = 2

torch.manual_seed(19)
rnn_cell = nn.RNNCell(input_size=n_features, hidden_size=hidden_dim)
rnn_state = rnn_cell.state_dict()

classifier = nn.Linear(hidden_dim, 1)


def get_single_loss(point):

    # Pytorch requires this format:  (num_layers * num_directions, batch_size, hidden_dim)
    hidden = torch.zeros(1, hidden_dim)

    X = torch.as_tensor(point).float()
    print("Input:", X.data)

    out = None
    for i in range(X.shape[0]):
        out = rnn_cell(X[i:i+1], hidden)
        print(f"Step {i}:, output:{out.data}, hidden:{hidden.data}")
        hidden = out

    # We will feed the last "out" to the classifier
    return classifier(out)



print(get_single_loss(points[0]))

# Well, there really is no point to doing just the training because you need back propagation to work.