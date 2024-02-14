# hidden is a tensor
# WHEN YOU CHANGE DATA TYPES, THE DEFAULT VALUES WILL CHANGE TOO!!j

import math
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


#
# We are having 2 hidden features which is why we have 2 rows in weights and bias
#

# Cell: book uses "ih"

weight_ih = torch.tensor([
    [ 0.3519, -0.6514],
    [ 0.3238,  0.5568]
], dtype=torch.float64, requires_grad=True)

bias_ih = torch.tensor([0.2198, 0.4712], dtype=torch.float64, requires_grad=True)

# Hidden

weight_hh = torch.tensor([
    [ 0.4279,  0.6832],
    [-0.4114,  0.5715]
], dtype=torch.float64, requires_grad=True)

bias_hh = torch.tensor([-0.4090, -0.1299], dtype=torch.float64, requires_grad=True)


## Don't touch the classifier, our focus is on RNN rather than classifier
classifier = nn.Linear(hidden_dim, 1)
classifier.weight.data = torch.tensor([[-0.2732, -0.1587]], dtype=torch.float64)
classifier.bias.data = torch.tensor([0.5806], dtype=torch.float64)
print('classifier:', classifier.state_dict())
# ALWAYS CHECK DEFAULT WEIGHTS. THEY MIGHT CHANGE AFTER YOU CHANGE DATA TYPES

loss = nn.BCEWithLogitsLoss()   # just sigmod with BCELoss

rnn_params = [weight_ih, weight_hh, bias_ih, bias_hh]
optimizer = optim.Adam(list(classifier.parameters()) + rnn_params, lr=0.01)

VERBOSE = False
# points, directions = points[:1], directions[:1]
EPOCH = 20
for epoch in range(EPOCH):

    ###################### Y_hat ######################
    classifier_outputs = []
    for i, point in enumerate(points):
        if VERBOSE:
            print("Input:", point)

        # Note this is different that 0_copy because it makes it easier
        # to reference. So I unwrapped it
        hidden = torch.tensor([0., 0.], requires_grad=True)
        if VERBOSE:
            print('initial hidden', hidden)

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

            # This doesn't work either!
            # hidden = [torch.tanh(pretan_x), torch.tanh(pretan_y)]

            # Needs to be in this shape so we can reuse it for the next run
            hidden = torch.stack([torch.tanh(pretan_x), torch.tanh(pretan_y)])

            if VERBOSE:
                print(f"Step {i}:, hidden:{hidden}")

        # No need to wrap and reinitialize hidden again,
        # just reshape it to the shape classifier expects
        hidden = hidden.unsqueeze(0)

        if VERBOSE:
            print('What are we feeding into the classifier?', hidden)
        temp = classifier(hidden)
        if VERBOSE:
            print('What comes out from the classifier?', temp)
        classifier_outputs.append(temp)

    ###################### end of Y_hat ######################


    # Classifier outputs becomes a list of tensor due to classifier
    classifier_outputs_tensor = torch.cat(classifier_outputs).view(-1).to(torch.float64)
    directions_tensor = torch.tensor(directions, dtype=torch.float64)

    # print('classifier_outputs_tensor', classifier_outputs)
    # print('directions_tensor', directions_tensor)

    training_loss = loss(classifier_outputs_tensor, directions_tensor)

    print(f"Epoch:{epoch}, training_loss:{training_loss.data}")

    # dot = make_dot(training_loss)
    # dot.render("DELETEME_2")
    # dot.view("DELETEME_2")  # This will open the saved pdf

    training_loss.backward()
    optimizer.step()
    optimizer.zero_grad()



