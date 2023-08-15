import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torch.nn.utils import rnn as rnn_utils

from StepByStep_v4 import StepByStep

##

import numpy as np

# def generate_sequences(n=128, variable_len=False, seed=13):
#     basic_corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
#     np.random.seed(seed)
#     bases = np.random.randint(4, size=n)
#     if variable_len:
#         lengths = np.random.randint(3, size=n) + 2
#     else:
#         lengths = [4] * n
#     directions = np.random.randint(2, size=n)
#     points = [basic_corners[[(b + i) % 4 for i in range(4)]][slice(None, None, d*2-1)][:l] + np.random.randn(l, 2) * 0.1 for b, d, l in zip(bases, directions, lengths)]
#     return points, directions

# ##

# points, directions = generate_sequences(n=128, seed=13)
# test_points, test_directions = generate_sequences(seed=19)

#with open('random_data.pickle', 'wb') as outf:
#    pickle.dump(dict(points=points, directions=directions, test_points=test_points, test_directions=test_directions), outf)

import pickle
with open('random_data.pickle', 'r') as inf:
    d = pickle.load(inf)

points, directions = d['points'], d['directions']
test_points, test_directions = d['test_points'], d['test_directions']

##

train_data = TensorDataset(
    torch.as_tensor(points).float(),
    torch.as_tensor(directions).view(-1, 1).float()
)

test_data = TensorDataset(
    torch.as_tensor(test_points).float(),
    torch.as_tensor(test_directions).view(-1, 1).float()
)

train_loader = DataLoader(
    train_data, batch_size=16, shuffle=True
)

test_loader = DataLoader(test_data, batch_size=16)

##

class SquareModel(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs):
        super(SquareModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.hidden = None
        # Simple RNN
        self.basic_rnn = nn.RNN(self.n_features, self.hidden_dim, batch_first=True)
        # Classifier to produce as many logits as outputs
        self.classifier = nn.Linear(self.hidden_dim, self.n_outputs)

    def forward(self, X):
        # X is batch first (N, L, F)
        # output is (N, L, H)
        # final hidden state is (1, N, H)
        batch_first_output, self.hidden = self.basic_rnn(X)

        # only last item in sequence (N, 1, H)
        last_output = batch_first_output[:, -1]
        # classifier will output (N, 1, n_outputs)
        out = self.classifier(last_output)

        # TODO: what are we feeding into classifier
        return out.view(-1, self.n_outputs)

##

torch.manual_seed(21)
model = SquareModel(n_features=2, hidden_dim=2, n_outputs=1)
loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)     # why ?

sbs_rnn = StepByStep(model, loss, optimizer)
sbs_rnn.set_loaders(train_loader, test_loader)
sbs_rnn.train(10)

fig = sbs_rnn.plot_losses()