import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Our data was in Numpy arrays, but we need to transform them
# into PyTorch's Tensors and then we send them to the
# chosen device
x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)

train_data = TensorDataset(x_train_tensor, y_train_tensor)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=16,
    shuffle=True
)
