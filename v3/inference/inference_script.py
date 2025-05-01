import torch
from v3.training.model.model import CBOWNeuralNetwork
import v3.training.training_utils as training_utils
import v3.inference.inference_utils as inference_utils


training_run_label: str = "nosubsampling"
epoch: int = 144
device: str = "cpu"
top: int = 3


print("load vocab-data...")

vocab_data = training_utils.load_vocab()

print("...loaded vocab-data")


print("load model-data...")

model_data = training_utils.load_model_data_at_epoch(
    training_run_label=training_run_label,
    epoch=epoch
)

print("model-data loaded...")


print("init model...")

model: CBOWNeuralNetwork = CBOWNeuralNetwork(
    vocab_size=vocab_data.vocab_size,
    hidden_layer_size=model_data.hidden_layer_size
).to(device)

model.load_state_dict(model_data.model_state_dict)

print("model inited...")


embeddings: torch.Tensor = model.input_to_hidden.weight.data


print("input: king - man + woman, expected: queen")
print(
    inference_utils.calc_embedding_a_minus_b_plus_c(
        word_a="king",
        word_b="man",
        word_c="woman",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)

print("input: boy - man + woman, expected: girl")
print(
    inference_utils.calc_embedding_a_minus_b_plus_c(
        word_a="boy",
        word_b="man",
        word_c="woman",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)

print("input: brother - man + woman, expected: sister")
print(
    inference_utils.calc_embedding_a_minus_b_plus_c(
        word_a="brother",
        word_b="man",
        word_c="woman",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)

print("input: berlin - paris + france, expected: germany")
print(
    inference_utils. calc_embedding_a_minus_b_plus_c(
        word_a="berlin",
        word_b="paris",
        word_c="france",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)

print("input: italian - french + france, expected: italy")
print(
    inference_utils. calc_embedding_a_minus_b_plus_c(
        word_a="italian",
        word_b="french",
        word_c="france",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)

print("input: fast - cold + hot, expected: slow")
print(
    inference_utils. calc_embedding_a_minus_b_plus_c(
        word_a="fast",
        word_b="cold",
        word_c="hot",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)


print("input: run - play + played, expected: ran")
print(
    inference_utils. calc_embedding_a_minus_b_plus_c(
        word_a="run",
        word_b="play",
        word_c="played",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)
