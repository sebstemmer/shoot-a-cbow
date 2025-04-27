import torch
from torch import nn
from torch.utils.data import DataLoader
import pickle
from v2.training.data_set import Dataset
from v2.training.model import CBOWNeuralNetwork
import sys
import v2.training.training_utils as training_utils
import collections
import v2.inference.inference_utils as inference_utils
import matplotlib.pyplot as plt

batch_size = 2048
hidden_layer_size = 300
num_epochs = 1000
learning_rate = 5
load_from_epoch = -1
subsampling_t = 1e-3

with open("./v2/data/processing/processed_data.pickle", "rb") as handle:
    processed_data = pickle.load(handle)

with open("./v2/data/processing/vocab_data.pickle", "rb") as handle:
    vocab_data = pickle.load(handle)


vocab: dict[str, tuple[int, int]] = vocab_data["vocab"]

vocab_frequencies: dict[int, float] = training_utils.create_vocab_frequencies(
    vocab
)

context_window_size: int = processed_data["context_window_size"]
vocab_size: int = processed_data["vocab_size"]
training_data: list[tuple[int, list[int]]] = processed_data["training_data"]

# calc entropy

print("calc entropy...")

ent_N = torch.zeros(vocab_size)
ent_nlogn = torch.zeros(vocab_size)

for td_idx, td in enumerate(training_data[0:1000000]):
    print(td_idx)
    all: list[int] = [td[0]] + td[1]

    # u = (torch.tensor(all) == torch.arange(  # type: ignore
    #    0, vocab_size).unsqueeze(1)).sum(dim=1)

    u = torch.bincount(torch.tensor(all), minlength=vocab_size)

    # torch.where(u > 0, u * torch.log((u+1e-10)), 0.0)
    ent_nlogn += u * torch.log((u+1e-10))
    ent_N += u

ent_N_save = torch.where(ent_N > 0, ent_N, 0)

ent = torch.where(
    ent_N_save > 0,
    -1.0 / ent_N_save * (
        ent_nlogn - torch.where(
            ent_N_save > 0, torch.log(
                ent_N_save
            ), 0.0) * ent_N_save
    ), 0.0)

top_values, top_indices = torch.topk(ent_N_save, k=30000)

# plt.hist(top_values)  # type: ignore
# plt.show()  # type: ignore

print(top_values[0:100])
print(top_indices[0:100])

print(top_values[2500:2600])
print(top_indices[2500:2600])

plt.plot(range(0, 30000), top_values)  # type: ignore
plt.show()  # type: ignore

idx_to_word_vocab = inference_utils.create_idx_to_word_vocab(
    vocab=vocab, vocab_size=vocab_size)

print([idx_to_word_vocab[top_val]
      for top_val in top_indices.tolist()])  # type: ignore

sys.exit()

print("there are " + str(len(training_data) / batch_size) + " batches")

device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


model = CBOWNeuralNetwork(
    vocab_size=vocab_size,
    hidden_layer_size=hidden_layer_size
).to(device)

print(model.input_to_hidden.weight.data.shape)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_fn = nn.CrossEntropyLoss()

if (load_from_epoch >= 0):
    checkpoint = torch.load(  # type: ignore
        "./v2/data/model/model_epoch_" + str(load_from_epoch) + ".pt")

    model.load_state_dict(checkpoint['model_state_dict'])

model.train()

for epoch in range(load_from_epoch + 1, num_epochs):
    print("epoch: " + str(epoch) + "\n")

    # subsampled_training_data = training_utils.subsample_training_data(
    #    training_data=training_data,
    #    vocab_frequencies=vocab_frequencies,
    #    subsampling_t=subsampling_t
    # )

    dataset = Dataset(
        training_data=training_data,
        context_window_size=context_window_size,
        vocab_size=vocab_size
    )

    dataLoader = DataLoader(  # type: ignore
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # print("in this epoch are " +
    #     str(len(subsampled_training_data) / batch_size) + " batches")

    print("in this epoch are " + str(len(training_data) / batch_size) + " batches")

    epoch_loss = 0.0
    counter = 0.0

    for (idx, batch) in enumerate(dataLoader):  # type: ignore
        optimizer.zero_grad()

        x, normed_mask, y = batch
        # print(idx)

        outputs = model(x.to(device), normed_mask.to(device))

        loss = loss_fn(outputs, y.to(device))
        epoch_loss += loss.item()
        counter += 1.0

        # print("loss: " + str(loss.item()))  # +
        # ", epoch loss: " + str(epoch_loss / counter))

        loss.backward()

        optimizer.step()  # type: ignore

        """ for name, param in model.named_parameters():
            print(name, param.grad.abs().mean())

        print(
            "Top-5 softmax probs:",
            torch.softmax(outputs[0], dim=-1).topk(5).values
        ) """

    print("epoch_loss")
    print(epoch_loss / counter)

    checkpoint_path = f"./v2/data/model/model_epoch_{epoch}.pt"
    torch.save({  # type: ignore
        "epoch": epoch,
        "hidden_layer_size": hidden_layer_size,
        "model_state_dict": model.state_dict(),
    },
        checkpoint_path
    )
    print(f"Checkpoint saved: {checkpoint_path}")
