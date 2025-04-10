import torch
from torch import nn

vocab_size = 12

training_data = [(1, [7, 8]), (2, [5, 0, 10, 11]), (7, [8, 9, 10])]

idx = 0

training_sample = training_data[idx]

cwt2 = max(
    [len(training_sample[1]) for training_sample in training_data]
)

X = torch.zeros(
    cwt2,
    vocab_size
)

X[:len(training_sample[1]), :] = nn.functional.one_hot(
    torch.tensor(training_sample[1]),
    vocab_size
)

print("X")
print(X)

print(training_sample[1])

print(
    nn.functional.one_hot(
        torch.tensor(training_sample[1]),
        vocab_size
    )
)

for idx_el, el in enumerate(training_sample[1]):
    X[idx_el, :] = nn.functional.one_hot(
        torch.tensor(el),
        vocab_size
    )

print(X)
