# 3 putting all the code into a nn.module to demonstrate understanding


### Directly copying from 2_unpack_success


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


VERBOSE = False

class RNN2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_RNN()
        self.init_classifier()

    def init_RNN(self):
        self.weight_ih = nn.Parameter(torch.tensor([
            [ 0.3519, -0.6514],
            [ 0.3238,  0.5568]
        ], dtype=torch.float64, requires_grad=True))

        self.bias_ih = nn.Parameter(torch.tensor(
            [0.2198, 0.4712],
            dtype=torch.float64, requires_grad=True))

        self.weight_hh = nn.Parameter(torch.tensor([
            [0.4279,  0.6832],
            [-0.4114,  0.5715]
        ], dtype=torch.float64, requires_grad=True))

        self.bias_hh = nn.Parameter(torch.tensor(
            [-0.4090, -0.1299],
        dtype=torch.float64, requires_grad=True))

    def init_classifier(self):
        ## Don't touch the classifier, our focus is on RNN rather than classifier
        self.classifier = nn.Linear(hidden_dim, 1)
        self.classifier.weight.data = torch.tensor([[-0.2732, -0.1587]], dtype=torch.float64)
        self.classifier.bias.data = torch.tensor([0.5806], dtype=torch.float64)
        print('classifier:', self.classifier.state_dict())
        # ALWAYS CHECK DEFAULT WEIGHTS. THEY MIGHT CHANGE AFTER YOU CHANGE DATA TYPES

    def forward(self, X):
        """Reads a list of 4 points and outputs the hidden state for classification

        X is a list of 4-corners
        [
            array([
                [ 1.03487506,  0.96613817],
                [ 0.80546093, -0.91690943],
                [-0.82507582, -0.94988627],
                [-0.86696831,  0.93424827]
            ]),
            array([
                [ 1.0184946 , -1.06510565],
                [ 0.88794931,  0.96533932],
                [-1.09113448,  0.92538647],
                [-1.07709685, -1.04139537]
            ]), ...
        """
        ret = []

        for corners in X:
            # This gets fed into the classifier, no need to assign it
            hidden = torch.tensor([0., 0.], requires_grad=False)
            if VERBOSE:
                print('initial hidden', hidden)

            for corner in corners:
                x_cord, y_cord = corner

                ix = x_cord * self.weight_ih[0][0] \
                    + y_cord * self.weight_ih[0][1] \
                    + self.bias_ih[0]

                iy = x_cord * self.weight_ih[1][0] \
                    + y_cord * self.weight_ih[1][1] \
                    + self.bias_ih[1]

                hx = hidden[0] * self.weight_hh[0][0] \
                    + hidden[1] * self.weight_hh[0][1] \
                    + self.bias_hh[0]

                hy = hidden[0] * self.weight_hh[1][0] \
                    + hidden[1] * self.weight_hh[1][1] \
                    + self.bias_hh[1]

                pretan_x = ix + hx
                pretan_y = iy + hy

                # Do NOT use math.tanh otherwise it won't be included in descent
                # hidden = [math.tanh(pretan_x), math.tanh(pretan_y)]

                # This doesn't work either!
                # hidden = [torch.tanh(pretan_x), torch.tanh(pretan_y)]

                # Needs to be in this shape so we can reuse it for the next run
                hidden = torch.stack([torch.tanh(pretan_x), torch.tanh(pretan_y)])

                if VERBOSE:
                    print(f"Step: hidden:{hidden}")

            # No need to wrap and reinitialize hidden again,
            # just reshape it to the shape classifier expects
            hidden = hidden.unsqueeze(0)
            temp = self.classifier(hidden)
            ret.append(temp)
            #print(temp)

        return ret







model = RNN2Classifier()

optimizer = optim.Adam(model.parameters(), lr=0.01)

loss = nn.BCEWithLogitsLoss()   # just sigmod with BCELoss


EPOCH = 20
for epoch in range(EPOCH):

    model.train()
    Y_hat = model(points)

    ## Copied directly from 2_unpack_success

    # Classifier outputs becomes a list of tensor due to classifier
    Y_hat_tensor = torch.cat(Y_hat).view(-1).to(torch.float64)
    Y_tensor = torch.tensor(directions, dtype=torch.float64)

    # print('classifier_outputs_tensor', classifier_outputs)
    # print('directions_tensor', directions_tensor)

    training_loss = loss(Y_hat_tensor, Y_tensor)

    print(f"Epoch:{epoch}, training_loss:{training_loss.data}")

    # dot = make_dot(training_loss)
    # dot.render("DELETEME_2")
    # dot.view("DELETEME_2")  # This will open the saved pdf

    training_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

