
# I think working out the partial derivative is too hard
# Perhaps you should use pytorch tensor to use the automatic gradient function in tensor

# - reimplement this using tensor with grad=True
#


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


input_w = torch.tensor([0.1, -0.2], requires_grad=True)
input_b = torch.tensor([0.1, 0.2], requires_grad=True)
hidden_w = torch.tensor([0.3, 0.4], requires_grad=True)
hidden_b = torch.tensor([0.3, 0.4], requires_grad=True)


EPOCH = 10
lr = 0.01
loss_fn = nn.BCEWithLogitsLoss()

classifier = nn.Linear(2, 1)

for epoch in range(EPOCH):
    pass


# def forward_one_sequence(hidden, sequence, name=""):
    # global input_w, input_b, hidden_w, hidden_b

# Initial hidden state is 0,0
hidden = torch.tensor([0.0, 0.0], requires_grad=True)

for x_idx, sequence in enumerate(d['points']):

    hidden_states = []

    for x0, x1 in sequence:
        x = torch.tensor([x0, x1], requires_grad=False)
        hidden = torch.tanh(x * input_w + input_b + hidden * hidden_w + hidden_b)

        logging.debug(f'sequence:{x_idx} hidden_x:{hidden}')

        hidden_states.append(hidden)

    # logging.debug(f'[input], w0:{input_w0:,.3f}, w1:{input_w1:,.3f}, b0:{input_b0:,.3f}, b1:{input_b1:,.3f}, '
                #   f'[hidden], w0:{hidden_w0:,.3f}, w1:{hidden_w1:,.3f}, b0:{hidden_b0:,.3f}, b1:{hidden_b1:,.3f}')

    # Last one is used for classifier
    yhat = classifier(hidden_states[-1])

    print('yhat', yhat)
    print("d['directions'][x_idx]", d['directions'][x_idx])

    loss = loss_fn(yhat, d['directions'][x_idx])

    # Compute gradient
    loss.backward()

    print('before', input_w)

    with torch.no_grad():
        input_w -= lr * input_w.grad

    print('after', input_w)


# hiddens = forward_one_sequence(hidden_0, d['points'][0], "test")

# make_dot(hiddens[0]).render("0", format="png")
# make_dot(hiddens[1]).render("1", format="png")
# make_dot(hiddens[2]).render("2", format="png")
# make_dot(hiddens[3]).render("3", format="png")


# Assuming 1 unroll for x0
# hidden_1 = math.tanh(temp)
#          = math.tanh(x * input_w + input_b + hidden_0 * hidden_w + hidden_b)
# so d_hidden_1 / d_input_w = derivative(math.tanh(x * input_w))
# too difficult...






