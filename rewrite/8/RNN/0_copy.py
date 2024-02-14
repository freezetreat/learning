

import torch.nn as nn
import torch
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
torch.set_default_dtype(torch.float64)


hidden_state = torch.zeros(2)

n_features = 2
hidden_dim = 2

rnn_cell = nn.RNNCell(input_size=n_features, hidden_size=hidden_dim)
rnn_state = rnn_cell.state_dict()
print('rnn cell', rnn_state)
# OrderedDict(
#     {'weight_ih': tensor([[ 0.3519, -0.6514], [ 0.3238,  0.5568]]),
#      'bias_ih': tensor([0.2198, 0.4712]),
#      'weight_hh': tensor([[ 0.4279,  0.6832], [-0.4114,  0.5715]]),
#      'bias_hh': tensor([-0.4090, -0.1299])})

## Don't touch the classifier, our focus is on RNN rather than classifier
classifier = nn.Linear(hidden_dim, 1)
classifier.weight.data = torch.tensor([[-0.2732, -0.1587]], dtype=torch.float64)
classifier.bias.data = torch.tensor([0.5806], dtype=torch.float64)
print('classifier:', classifier.state_dict())
# classifier: OrderedDict({'weight': tensor([[-0.2732, -0.1587]]), 'bias': tensor([0.5806])})

# ALWAYS CHECK DEFAULT WEIGHTS. THEY MIGHT CHANGE AFTER YOU CHANGE DATA TYPES

EPOCH = 20
# points, directions = points[:1], directions[:1]


# loss = nn.BCELoss()           # expects number from 0 to 1
loss = nn.BCEWithLogitsLoss()   # just sigmod with BCELoss


# TODO: why are we just optimizing the rnn_cell parameters and
# and not the parameters of our classifier?
# optimizer = optim.Adam(rnn_cell.parameters(), lr=0.01)
# Answer: ChatGPT to the rescue!!!!
optimizer = optim.Adam(list(rnn_cell.parameters()) + list(classifier.parameters()), lr=0.01)

VERBOSE = False

for epoch in range(EPOCH):

    ###################### Y_hat ######################
    classifier_outputs = []
    for i, point in enumerate(points):
        # Pytorch requires this format:  (num_layers * num_directions, batch_size, hidden_dim)
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


