
import torch
import pickle
from v2.training.model import CBOWNeuralNetwork
import v2.inference.inference_utils as inference_utils
from typing import Union, Any

torch_load = torch.load  # type: ignore


epoch = 80

device = "cpu"

with open("./v2/data/processing/vocab_data.pickle", "rb") as handle:
    vocab_data = pickle.load(handle)

vocab_size = vocab_data["vocab_size"]
vocab = vocab_data["vocab"]

word_to_idx_vocab: dict[str, int] = inference_utils.create_word_to_idx_vocab(
    vocab=vocab,
    vocab_size=vocab_size
)

idx_to_word_vocab: dict[int, str] = inference_utils.create_idx_to_word_vocab(
    vocab=vocab,
    vocab_size=vocab_size
)

checkpoint = torch_load(
    "./v2/data/model/model_epoch_" + str(epoch) + ".pt"
)

hidden_layer_size: int = checkpoint["hidden_layer_size"]

model: CBOWNeuralNetwork = CBOWNeuralNetwork(
    vocab_size=vocab_size,
    hidden_layer_size=hidden_layer_size
).to(device)

embeddings: torch.Tensor = model.input_to_hidden.weight.data

model.load_state_dict(checkpoint["model_state_dict"])


def getVecFromWord(word: str):
    return model.input_to_hidden.weight.data[word_to_idx_vocab[word], :]


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

# v3: torch.Tensor = model.input_to_hidden.weight.data[word_to_idx_vocab["man"], :]

print("v3.shape")
print(v3.shape)
print("model.input_to_hidden.weight.data.shape")
print(model.input_to_hidden.weight.data.shape)

sim = inference_utils.cosine_similarity_in_last_dim(
    vec=v3.unsqueeze(0),
    embeddings=model.input_to_hidden.weight.data,
    excludes=[word_to_idx_vocab["king"],
              word_to_idx_vocab["woman"], word_to_idx_vocab["man"]]
)

print("sim.shape)")
print(sim.shape)

top_vals, top_idxs = torch.topk(sim, k=5)
print(top_vals)
print(top_idxs)

idx_list: list[int] = top_idxs.tolist()  # type: ignore

print([idx_to_word_vocab[idx] for idx in idx_list])

print(
    inference_utils. check_relation(
        word_a="king",
        word_b="man",
        word_c="woman",
        embeddings=embeddings,
        word_to_idx_vocab=word_to_idx_vocab,
        idx_to_word_vocab=idx_to_word_vocab,
        top=3
    )
)

print(
    inference_utils. check_relation(
        word_a="boy",
        word_b="man",
        word_c="woman",
        embeddings=embeddings,
        word_to_idx_vocab=word_to_idx_vocab,
        idx_to_word_vocab=idx_to_word_vocab,
        top=3
    )
)

print(
    inference_utils. check_relation(
        word_a="dog",
        word_b="man",
        word_c="men",
        embeddings=embeddings,
        word_to_idx_vocab=word_to_idx_vocab,
        idx_to_word_vocab=idx_to_word_vocab,
        top=3
    )
)

print(
    inference_utils. check_relation(
        word_a="berlin",
        word_b="paris",
        word_c="france",
        embeddings=embeddings,
        word_to_idx_vocab=word_to_idx_vocab,
        idx_to_word_vocab=idx_to_word_vocab,
        top=3
    )
)

print(
    inference_utils. check_relation(
        word_a="paris",
        word_b="miami",
        word_c="florida",
        embeddings=embeddings,
        word_to_idx_vocab=word_to_idx_vocab,
        idx_to_word_vocab=idx_to_word_vocab,
        top=3
    )
)

print(
    inference_utils. check_relation(
        word_a="german",
        word_b="french",
        word_c="france",
        embeddings=embeddings,
        word_to_idx_vocab=word_to_idx_vocab,
        idx_to_word_vocab=idx_to_word_vocab,
        top=3
    )
)

print(
    inference_utils. check_relation(
        word_a="italy",
        word_b="germany",
        word_c="hitler",
        embeddings=embeddings,
        word_to_idx_vocab=word_to_idx_vocab,
        idx_to_word_vocab=idx_to_word_vocab,
        top=3
    )
)


print(
    inference_utils. check_relation(
        word_a="fast",
        word_b="cold",
        word_c="hot",
        embeddings=embeddings,
        word_to_idx_vocab=word_to_idx_vocab,
        idx_to_word_vocab=idx_to_word_vocab,
        top=3
    )
)
