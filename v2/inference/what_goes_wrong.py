
import torch
import pickle
from v2.training.model import CBOWNeuralNetwork

epoch = 16

device = "cpu"

with open("./v2/data/processing/vocab_data.pickle", "rb") as handle:
    vocab_data = pickle.load(handle)

vocab_size = vocab_data["vocab_size"]
vocab = vocab_data["vocab"]
indexed_vocab = {
    word: word_count_and_idx[1] for word,
    word_count_and_idx in vocab.items() if word_count_and_idx[1] < vocab_size
}


checkpoint = torch.load(  # type: ignore
    "./v2/data/model/model_epoch_" + str(epoch) + ".pt"
)
hidden_layer_size = checkpoint["hidden_layer_size"]

model = CBOWNeuralNetwork(
    vocab_size=vocab_size,
    hidden_layer_size=hidden_layer_size
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])

print(model.input_to_hidden.weight.data.shape)

a = model.input_to_hidden.weight.data

print(a[1, :].shape)
print(a[1, :].mean())
print(a.mean(dim=1)[1])
print(a.std(dim=1)[1])
print(a.mean(dim=1)[2])
print(a.mean(dim=1)[10])
print(a.mean(dim=1)[30])


print(torch.topk(input=a.std(dim=1), k=5))
