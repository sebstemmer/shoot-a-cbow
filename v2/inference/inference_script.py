
import torch
import pickle
from v2.training.model import CBOWNeuralNetwork

epoch = 10

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


def getVecFromWord(word: str):
    return model.input_to_hidden.weight.data[indexed_vocab[word], :]


def getVecFromWordIdx(wordIdx: int):
    return model.input_to_hidden.weight.data[wordIdx, :]


def cos(aWord: str, bWord: str):
    return torch.nn.functional.cosine_similarity(
        getVecFromWord(aWord),
        getVecFromWord(bWord),
        dim=0
    )


def cosIdx(aIdx: int, bIdx: int):
    return torch.nn.functional.cosine_similarity(
        getVecFromWordIdx(aIdx),
        getVecFromWordIdx(bIdx),
        dim=0
    )


manMinusWoman = getVecFromWord("man") - getVecFromWord("woman")

kingMinusQueen = getVecFromWord("king") - getVecFromWord("queen")

print("cos(man, king)")
print(
    torch.nn.functional.cosine_similarity(
        manMinusWoman,
        kingMinusQueen,
        dim=0
    )
)

v3 = getVecFromWord("king") - getVecFromWord("man") + getVecFromWord("woman")

# v3 = model.input_to_hidden.weight.data[v2, :] - \
#   model.input_to_hidden.weight.data[v1, :] + \
#   model.input_to_hidden.weight.data[v0, :]

v3 = model.input_to_hidden.weight.data[indexed_vocab["man"], :]

sim = torch.zeros(vocab_size)

for v in indexed_vocab.values():
    # print(v)
    sim[v] = torch.nn.functional.cosine_similarity(
        v3,
        model.input_to_hidden.weight.data[v, :],
        dim=0
    )

print(sim.shape)

top_vals, top_idxs = torch.topk(sim, k=10)

keys = [(k, sim[v]) for k, v in indexed_vocab.items() if v in top_idxs]

print(keys)
