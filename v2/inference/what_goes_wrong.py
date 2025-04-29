
import torch
import pickle
from v3.training.model.model import CBOWNeuralNetwork
import v2.training.training_utils as training_utils
import v2.inference.inference_utils as inference_utils
import matplotlib.pyplot as plt
import sys
import v2.utils.utils as utils
import random

epoch = 16

subsampling_t = 1e-3

device = "cpu"

print("loading processed_data...")

with open("./v2/data/processing/processed_data.pickle", "rb") as handle:
    processed_data = pickle.load(handle)

print("...loaded processed_data...")

print("loading vocab_data...")

with open("./v2/data/processing/vocab_data.pickle", "rb") as handle:
    vocab_data = pickle.load(handle)

print("...loaded vocab_data...")

training_data: list[tuple[int, list[int]]] = processed_data["training_data"]


vocab: dict[str, tuple[int, int]] = vocab_data["vocab"]

print(list(vocab.keys())[2000:2100])

sys.exit()

vocab_size: int = vocab_data["vocab_size"]

vocab_frequencies: dict[int, float] = training_utils.create_vocab_frequencies(
    vocab
)

idx_to_word_vocab: dict[int, str] = inference_utils.create_idx_to_word_vocab(
    vocab=vocab,
    vocab_size=vocab_size
)

print("subsample training_data...")

subsampled_training_data: list[tuple[int, list[int]]] = training_utils.subsample_training_data(
    training_data=training_data,
    vocab_frequencies=vocab_frequencies,
    subsampling_t=subsampling_t
)

print("...subsampled training_data")

samples: int = 50
start_idx: int = random.randint(0, len(subsampled_training_data) - 1 - samples)
end_idx: int = start_idx + samples

for td in subsampled_training_data[start_idx:end_idx]:
    print(
        "target word: " + idx_to_word_vocab[td[0]] + ", context words: " + ", ".join(
            [idx_to_word_vocab[context_idx] for context_idx in td[1]]
        )
    )

sys.exit()

all_idx: list[int] = utils.flatten(
    utils.flatten(
        [[
            [td[0]],
            td[1]
        ] for td in subsampled_training_data]
    )
)

# random.shuffle(all_idx)
all_idx = all_idx[0:50000]

total_idx = len(all_idx)

for i in range(0, 100):
    q = [idx for idx in all_idx if idx == i]
    print("idx " + str(i) + ", relation: " + str(len(q)/total_idx))

for l in range(1, 20):
    r = [
        td for td in subsampled_training_data if len(td[1]) == l
    ]
    print("l " + str(l) + " " + str(len(r)))

plt.hist(  # type: ignore
    [len(td[1]) for td in subsampled_training_data]
)

# a = [
#    td for td in len1 if td[1][0] == 1
# ]

# print(len(a))
print(len(subsampled_training_data))

# plt.hist(  # type: ignore
#    [td[1][0] for td in subsampled_training_data if len(td[1]) == 1]
# )
plt.show()  # type: ignore

sys.exit()

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
