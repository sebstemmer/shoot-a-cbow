import torch
from v2.training.data_set import Dataset

training_data = [(0, [1, 2, 3]), (2, [0, 4, 5]), (4, [1, 2])]
idx = 1
vocab_size = 6
context_window_size = 2

dataset = Dataset(
    training_data=training_data,
    context_window_size=context_window_size,
    vocab_size=vocab_size
)

X, mask, Y = dataset.__getitem__(idx)

assert torch.allclose(
    X,
    torch.tensor([0, 4, 5, 6])
)

assert torch.allclose(
    mask,
    torch.tensor([
        [[1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0]]
    ])
)

assert torch.allclose(
    Y, torch.tensor(2)
)

assert dataset.__len__() == 3
