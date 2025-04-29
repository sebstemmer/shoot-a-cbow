import torch
from torch import nn
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
import v3.preprocessing.preprocessing_utils as preprocessing_utils
import v3.training.training_utils as training_utils

batch_size = 2048
hidden_layer_size = 300
num_epochs = 1000
learning_rate = 5
load_from_epoch = -1
subsampling_t = 1e-3
subsampling_pow = 1


print("load preprocessed-data...")

with open(preprocessing_utils.path_to_data_folder + "preprocessing/preprocessed_data.pickle", "rb") as handle:
    preprocessed_data: preprocessing_utils.PreprocessedData = pickle.load(
        handle
    )

print("...preprocessed-data loaded")


print("load vocab...")

with open(preprocessing_utils.path_to_data_folder + "preprocessing/vocab.pickle", "rb") as handle:
    vocab: preprocessing_utils.Vocab = pickle.load(handle)

print("...vocab loaded")


device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"
print(f"using {device} device")


print("init model, optimizer and loss-function...")

model: training_utils.CBOWNeuralNetwork = training_utils.CBOWNeuralNetwork(
    vocab_size=vocab.vocab_size,
    hidden_layer_size=hidden_layer_size
).to(device)

optimizer: torch.optim.SGD = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)

cross_entropy_loss_function: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

print("...inited model, optimizer and loss-function")


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

        loss = cross_entropy_loss_function(outputs, y.to(device))
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
