import pickle
import cbow_functions as cf
import os
import torch
from torch import nn
import sys
from torch.utils.data import DataLoader

# with open('preprocessed_text.pickle', 'rb') as handle:
#     sentences_with_words = pickle.load(handle)

# vocab = cf.create_vocab(sentences_with_words)

# vocab_size = list(vocab.values())[-1]

# sentences = [[vocab[word] for word in sentence]
#              for sentence in sentences_with_words]
# training_data = cf.create_training_data(sentences, 2)

# print(training_data[0:2])


vocab_size = 12
training_data = [(1, [7, 8]), (2, [5, 0, 10, 11]), (7, [8, 9, 10])]

print("hi...\n")

print(len(training_data))
print((max([len(training_sample[1]) for training_sample in training_data])))
print(vocab_size)

X = torch.zeros(
    len(training_data),
    (max([len(training_sample[1]) for training_sample in training_data])),
    vocab_size
)

print("create X...\n")

for idx, training_sample in enumerate(training_data):
    for idx_el, el in enumerate(training_sample[1]):
        X[idx, idx_el, :] = nn.functional.one_hot(
            torch.tensor(el),
            vocab_size
        )

print("...X created\n\n")

CS = torch.zeros(
    len(training_data),
    dtype=torch.short
)

for idx, training_sample in enumerate(training_data):
    CS[idx] = len(training_sample[1])

mask = (torch.arange(0, X.shape[1]).unsqueeze(0) < CS.unsqueeze(1)).float()

normed_mask = mask / mask.sum(dim=1, keepdim=True)
normed_mask = normed_mask.unsqueeze(1)


class CBowDataset(torch.utils.data.Dataset):
    def __init__(self, training_data, vocab_size):
        self.training_data = training_data
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        training_sample = self.training_data[idx]

        X = torch.zeros(
            (max([len(training_sample[1])
             for training_sample in training_data])),
            self.vocab_size
        )

        for idx_el, el in enumerate(training_sample[1]):
            X[idx_el, :] = nn.functional.one_hot(
                torch.tensor(el),
                self.vocab_size
            )

        CS = torch.tensor(len(training_sample[1]), dtype=torch.short)

        mask = (
            torch.arange(0, X.shape[0]) < CS
        ).float()

        normed_mask = mask / mask.sum()
        normed_mask = normed_mask

        return X, normed_mask


dataset = CBowDataset(training_data, vocab_size)
train_dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

data_iter = iter(train_dataloader)
x, cs = next(data_iter)

print("Image batch shape:", x.shape)
print("Label batch shape:", cs.shape)
print("Image dtype:", x)
print("Label dtype:", cs)

device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

hidden_layer_size = 5


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(
            vocab_size,
            hidden_layer_size,
            bias=True
        )

    def forward(self, x, normed_mask):
        y = self.linear(x)

        print(normed_mask.shape)
        print(y.shape)

        masked_y = torch.matmul(normed_mask, y)

        return masked_y.squeeze(1)

        # mean = torch.zeros(
        #    x.shape[0], x.shape[2]
        # )
        # for idx, c in enumerate(cs):
        #    mean[idx] = torch.mean(x[idx, 0:c.item(), :], dim=0)


model = NeuralNetwork().to(device)

result = model(X.to(device), normed_mask.to(device))

print(result.shape)

# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")

# print(f"Model structure: {model}\n\n")

# for name, param in model.named_parameters():
#    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
