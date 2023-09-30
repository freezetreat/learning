

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



### RNN


torch.manual_seed(19)

hidden_state = torch.zeros(2)

n_features = 2
hidden_dim = 2

rnn_cell = nn.RNNCell(input_size=n_features, hidden_size=hidden_dim)
rnn_state = rnn_cell.state_dict()

classifier = nn.Linear(hidden_dim, 1)

EPOCH = 100
# points, directions = points[:2], directions[:2]


# loss = nn.BCELoss()           # expects number from 0 to 1
loss = nn.BCEWithLogitsLoss()   # just sigmod with BCELoss


# TODO: why are we just optimizing the rnn_cell parameters and
# and not the parameters of our classifier?
# optimizer = optim.Adam(rnn_cell.parameters(), lr=0.01)
# Answer: ChatGPT to the rescue!!!!
optimizer = optim.Adam(list(rnn_cell.parameters()) + list(classifier.parameters()), lr=0.01)



for epoch in range(EPOCH):

    ###################### Y_hat ######################
    classifier_outputs = []
    for i, point in enumerate(points):
        # Pytorch requires this format:  (num_layers * num_directions, batch_size, hidden_dim)
        hidden = torch.zeros(1, hidden_dim)

        X = torch.as_tensor(point).float()
        # print("Input:", X.data)

        out = None
        for i in range(X.shape[0]):
            out = rnn_cell(X[i:i+1], hidden)
            # print(f"Step {i}:, output:{out.data}, hidden:{hidden.data}")
            hidden = out

        # We will feed the last "out" to the classifier
        classifier_outputs.append(classifier(out))

    ###################### end of Y_hat ######################

    # Convert [ tensor, .. ] to tensor([float, ...])
    # print("Classifier (Before)", classifier_outputs)
    classifier_outputs_tensor = torch.cat(classifier_outputs).view(-1).to(torch.float64)
    # print("Classifier (After)", classifier_outputs_tensor)

    # Convert directions numpy array to a PyTorch tensor
    # print("Directions (Before)", directions)
    directions_tensor = torch.tensor(directions, dtype=torch.float64)
    # print("Directions (After)", directions_tensor)

    # Now we need to compute loss
    training_loss = loss(classifier_outputs_tensor, directions_tensor)

    training_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch:{epoch}, training_loss:{training_loss.data}")

