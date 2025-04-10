import cbow_functions as cf
import torch
from torch import nn
import sys
from torch.utils.data import DataLoader
import pickle

with open('preprocessed_text.pickle', 'rb') as handle:
    sentences_with_words = pickle.load(handle)

vocab = cf.create_vocab(sentences_with_words)

vocab_size = list(vocab.values())[-1] + 1

sentences = [
    [vocab[word] for word in sentence] for sentence in sentences_with_words
]

context_window_size = 4

training_data = cf.create_training_data(
    sentences=sentences,
    context_window_size=context_window_size
)

batch_size = 512
smaller_training_data = training_data[0:100 * batch_size]

# print(training_data[0:2])


# vocab_size = 12
# training_data = [(1, [7, 8]), (2, [5, 0, 10, 11]), (7, [8, 9, 10])]


class CBowDataset(torch.utils.data.Dataset):
    def __init__(self, training_data, context_window_size, vocab_size):
        self.training_data = training_data
        self.context_window_size = context_window_size
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        # print("retrieve data idx " + str(idx) + "...")
        training_sample = self.training_data[idx]

        X = torch.zeros(
            self.context_window_size * 2 + 1,
            self.vocab_size
        )

        X[:len(training_sample[1]), :] = nn.functional.one_hot(
            torch.tensor(training_sample[1]),
            vocab_size
        )

        CS = torch.tensor(len(training_sample[1]), dtype=torch.short)

        mask = (
            torch.arange(0, X.shape[0]) < CS
        ).float()

        normed_mask = mask / mask.sum()
        normed_mask = normed_mask.unsqueeze(0)

        Y = torch.tensor(training_sample[0])

        return X, normed_mask, Y


dataset = CBowDataset(
    training_data=smaller_training_data,
    context_window_size=context_window_size,
    vocab_size=vocab_size
)

device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

hidden_layer_size = 300


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden = nn.Linear(
            vocab_size,
            hidden_layer_size,
            bias=True
        )
        self.hidden_to_output = nn.Linear(
            hidden_layer_size,
            vocab_size,
            bias=True
        )

    def forward(self, x, normed_mask):
        hidden = self.input_to_hidden(x)

        masked_hidden = torch.matmul(normed_mask, hidden)

        output = self.hidden_to_output(masked_hidden.squeeze(1))

        return output


model = NeuralNetwork().to(device)

# result = model(x.to(device), cs.to(device))

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

print("num batches: " + str(len(smaller_training_data) / batch_size))

num_epochs = 10
learning_rate = 0.5

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_fn = nn.CrossEntropyLoss()

model.train()

for epoch in range(0, num_epochs):
    print("epoch: " + str(epoch) + "\n")
    for batch in dataloader:
        optimizer.zero_grad()

        x, normed_mask, y = batch

        outputs = model(x.to(device), normed_mask.to(device))

        loss = loss_fn(outputs, y.to(device))

        print("loss: " + str(loss.item()))

        loss.backward()

        optimizer.step()

    checkpoint_path = f"model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")

# print(f"Model structure: {model}\n\n")

# for name, param in model.named_parameters():
#    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
