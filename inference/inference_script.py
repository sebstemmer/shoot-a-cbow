import torch
from training.model.model import CBOWNeuralNetwork
import training.training_utils as training_utils
import inference.inference_utils as inference_utils


preprocessing_run_label: str = "vs_30_cw_4"
training_run_label: str = "vs_30_cw_4_hl_300_noss"
epoch: int = 19
top: int = 5


print("load vocab-data...")

vocab_data = training_utils.load_vocab(
    preprocessing_run_label=preprocessing_run_label
)

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
).to("cpu")

model.load_state_dict(model_data.model_state_dict)

print("model inited...")


embeddings: torch.Tensor = model.input_to_hidden.weight.data


print("similar to: germany")
print(
    inference_utils.get_similar_words(
        word="germany",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)


print("similar to: car")
print(
    inference_utils.get_similar_words(
        word="car",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)

print("similiar to: cat")
print(
    inference_utils.get_similar_words(
        word="cat",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)

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
    inference_utils.calc_embedding_a_minus_b_plus_c(
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
    inference_utils.calc_embedding_a_minus_b_plus_c(
        word_a="italian",
        word_b="french",
        word_c="france",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)

print("input: small - hot + cold, expected: large")
print(
    inference_utils.calc_embedding_a_minus_b_plus_c(
        word_a="small",
        word_b="hot",
        word_c="cold",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)


print("input: hot - fast + slow, expected: cold")
print(
    inference_utils.calc_embedding_a_minus_b_plus_c(
        word_a="hot",
        word_b="fast",
        word_c="slow",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)


print("input: run - play + played, expected: ran")
print(
    inference_utils.calc_embedding_a_minus_b_plus_c(
        word_a="run",
        word_b="play",
        word_c="played",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)


print("input: listen - present + presented, expected: listened")
print(
    inference_utils.calc_embedding_a_minus_b_plus_c(
        word_a="listen",
        word_b="present",
        word_c="presented",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)


print("input: car - country + countries, expected: cars")
print(
    inference_utils.calc_embedding_a_minus_b_plus_c(
        word_a="car",
        word_b="country",
        word_c="countries",
        embeddings=embeddings,
        word_to_idx_vocab=vocab_data.word_to_idx_vocab,
        idx_to_word_vocab=vocab_data.idx_to_word_vocab,
        top=top
    )
)
