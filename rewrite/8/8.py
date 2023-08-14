
# I think working out the partial derivative is too hard
# Perhaps you should use pytorch tensor to use the automatic gradient function in tensor

# - reimplement this using tensor with grad=True
#



import math
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

import pickle
with open('random_data.pickle', 'rb') as inf:
    d = pickle.load(inf)

points, directions = d['points'], d['directions']
test_points, test_directions = d['test_points'], d['test_directions']


# SquareModel(n_features=2, hidden_dim=2, n_outputs=1)

# coefficients
input_w0, input_w1 = 0, 0
hidden_w0, hidden_w1 = 0, 0

input_w0, input_w1 = 0.1, -0.2
input_b0, input_b1 = 0.1, 0.2
hidden_w0, hidden_w1 = 0.3, 0.4
hidden_b0, hidden_b1 = 0.3, 0.4


def forward_one_sequence(sequence, name=""):
    global hidden_w0, hidden_w1, hidden_b0, hidden_b1, \
        input_w0, input_w1, input_b0, input_b1

    output = []

    hidden_x0, hidden_x1 = 0, 0
    for x0, x1 in sequence:
        temp_0 = x0 * input_w0 + input_b0 + hidden_x0 * hidden_w0 + hidden_b0
        temp_1 = x1 * input_w1 + input_b1 + hidden_x1 * hidden_w1 + hidden_b1
        temp_0 = math.tanh(temp_0)
        temp_1 = math.tanh(temp_1)

        # update hidden state
        hidden_x0, hidden_x1 = temp_0, temp_1

        logging.debug(f'sequence:{name} '
                      f'hidden_x:{hidden_x0:,.3f}, {hidden_x1:,.3f}')

        # Output the hidden state
        output.append((hidden_x0, hidden_x1))

    logging.debug(f'[input], w0:{input_w0:,.3f}, w1:{input_w1:,.3f}, b0:{input_b0:,.3f}, b1:{input_b1:,.3f}, '
                  f'[hidden], w0:{hidden_w0:,.3f}, w1:{hidden_w1:,.3f}, b0:{hidden_b0:,.3f}, b1:{hidden_b1:,.3f}')

    return output



print(forward_one_sequence(d['points'][0], "test"))


# Assuming 1 unroll for x0
# hidden_1 = math.tanh(temp)
#          = math.tanh(x * input_w + input_b + hidden_0 * hidden_w + hidden_b)
# so d_hidden_1 / d_input_w = derivative(math.tanh(x * input_w))
# too difficult...






