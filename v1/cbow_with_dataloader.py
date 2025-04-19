import cbow_functions as cf
import torch
from torch import nn
import sys
from torch.utils.data import DataLoader
import pickle
from cbow_neural_network.cbow_neural_network import CBOWNeuralNetwork
from cbow_data_set.cbow_data_set import CBOWDataset
import random

# hyperparameters

context_window_size = 2
batch_size = 2048
num_batches = 100
hidden_layer_size = 300
num_epochs = 1000
learning_rate = 50
context_window_training_data_threshold = 1

with open("data/preprocessed_text.pickle", "rb") as handle:
    sentences_with_words = pickle.load(handle)

vocab = cf.create_vocab(sentences_with_words)

vocab_size = list(vocab.values())[-1] + 1

sentences = [
    [vocab[word] for word in sentence] for sentence in sentences_with_words
]

training_data = cf.create_training_data(
    sentences=sentences,
    context_window_size=context_window_size
)

training_data_with_context_window_threshold = [
    td for td in training_data if len(td[1]) >= context_window_training_data_threshold * 2
]

smaller_training_data = training_data_with_context_window_threshold


random.shuffle(smaller_training_data)

print("smaller_training_data.shape")
print(len(smaller_training_data))


print("numBatches")
print(len(smaller_training_data) / batch_size)

print(smaller_training_data[0:5])

# print(training_data[0:2])


# vocab_size = 12
# training_data = [(1, [7, 8]), (2, [5, 0, 10, 11]), (7, [8, 9, 10])]


dataset = CBOWDataset(
    training_data=smaller_training_data,
    context_window_size=context_window_size,
    vocab_size=vocab_size
)

device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


model = CBOWNeuralNetwork(
    vocab_size=vocab_size,
    hidden_layer_size=hidden_layer_size
).to(device)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True
)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_fn = nn.CrossEntropyLoss()

load_from_epoch = 11

if (load_from_epoch >= 0):
    checkpoint = torch.load("data/model_epoch_" + str(load_from_epoch) + ".pt")

    model.load_state_dict(checkpoint['model_state_dict'])

model.train()

for epoch in range(load_from_epoch + 1, num_epochs):
    print("epoch: " + str(epoch) + "\n")
    epoch_loss = 0.0
    counter = 0.0
    for (idx, batch) in enumerate(dataloader):
        optimizer.zero_grad()

        x, normed_mask, y = batch
        print(idx)

        outputs = model(x.to(device), normed_mask.to(device))

        loss = loss_fn(outputs, y.to(device))
        epoch_loss += loss.item()
        counter += 1.0

        print("loss: " + str(loss.item()) +
              ", epoch loss: " + str(epoch_loss / counter))

        loss.backward()

        optimizer.step()

        # for name, param in model.named_parameters():
        #    print(name, param.grad.abs().mean())

    # print(
    #    "Top-5 softmax probs:",
    #    torch.softmax(outputs[0], dim=-1).topk(5).values
    # )

    print("epoch_loss")
    print(epoch_loss / counter)

    checkpoint_path = f"data/model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
