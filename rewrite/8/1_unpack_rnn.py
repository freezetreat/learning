# Unpacking RNN cell to raw numbers
# Note that this is meant to fail because hidden is NOT a tensor
# so when you do gradient descent you are missing out a bunch of shit

import math
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


#
# We are having 2 hidden features which is why we have 2 rows in weights and bias
#

# Cell: book uses "ih"

weight_ih = torch.tensor([
    [0.6627, -0.4245],
    [ 0.5373,  0.2294]
], dtype=torch.float64, requires_grad=True)

bias_ih = torch.tensor([0.4954, 0.6533], dtype=torch.float64, requires_grad=True)

# Hidden

weight_hh = torch.tensor([
    [-0.4015, -0.5385], [-0.1956, -0.6835]
], dtype=torch.float64, requires_grad=True)

bias_hh = torch.tensor([-0.3565, -0.2904], dtype=torch.float64, requires_grad=True)


## Don't touch the classifier, our focus is on RNN rather than classifier
classifier = nn.Linear(hidden_dim, 1)

loss = nn.BCEWithLogitsLoss()   # just sigmod with BCELoss

rnn_params = [weight_ih, weight_hh, bias_ih, bias_hh]
optimizer = optim.Adam(list(classifier.parameters()) + rnn_params, lr=0.01)


points, directions = points[:1], directions[:1]
EPOCH = 1
for epoch in range(EPOCH):

    ###################### Y_hat ######################
    classifier_outputs = []
    for i, point in enumerate(points):
        # print("Input:", point)

        # Note this is different that 0_copy because it makes it easier
        # to reference. So I unwrapped it
        hidden = [0, 0]

        for i in range(4):          # 4 sides of a square
            x_cord, y_cord = point[i]

            ix = x_cord * weight_ih[0][0] \
                + y_cord * weight_ih[0][1] \
                + bias_ih[0]

            iy = x_cord * weight_ih[1][0] \
                + y_cord * weight_ih[1][1] \
                + bias_ih[1]

            hx = hidden[0] * weight_hh[0][0] \
                + hidden[1] * weight_hh[0][1] \
                + bias_hh[0]

            hy = hidden[0] * weight_hh[1][0] \
                + hidden[1] * weight_hh[1][1] \
                + bias_hh[1]

            pretan_x = ix + hx
            pretan_y = iy + hy

            # Do NOT use math.tanh otherwise it won't be included in descent
            # hidden = [math.tanh(pretan_x), math.tanh(pretan_y)]
            hidden = [torch.tanh(pretan_x), torch.tanh(pretan_y)]
            print(f"Step {i}:, hidden:{hidden}")

        # Pass the hidden state to our classifier
        # note that we must wrap hidden inside a list because we unwrapped
        # hidden from above. (should start with hidden = [[0, 0]] instead)
        hidden = torch.tensor([hidden], requires_grad=True)
        print('What are we feeding into the classifier?', hidden)
        classifier_outputs.append(classifier(hidden))

    ###################### end of Y_hat ######################

    # Classifier outputs becomes a list of tensor due to classifier
    classifier_outputs_tensor = torch.cat(classifier_outputs).view(-1).to(torch.float64)
    directions_tensor = torch.tensor(directions, dtype=torch.float64)

    # print('classifier_outputs_tensor', classifier_outputs)
    # print('directions_tensor', directions_tensor)

    training_loss = loss(classifier_outputs_tensor, directions_tensor)

    dot = make_dot(training_loss)
    dot.render("DELETEME_0_copy")
    dot.view("DELETEME_0_copy")  # This will open the saved pdf

    training_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch:{epoch}, training_loss:{training_loss.data}")

