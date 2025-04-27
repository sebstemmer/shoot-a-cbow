import v2.data_processing.data_processing_utils as utils
import random
import pickle
from typing import Union

n_most_frequent_words: int = 30000
context_window_size: int = 4

raw_text: str = open("./v2/data/raw/AllCombined.txt", "r").read()

sentences: list[str] = utils.split_into_sentences(raw_text)

sentences: list[str] = utils.remove_punctuation(sentences)

sentences: list[str] = utils.lower(sentences)

sentences_in_words: list[list[str]] = utils.split_into_words(sentences)

vocab: dict[str, tuple[int, int]] = utils.count_words_and_create_indices(
    sentences_in_words
)

sentences_in_idxs: list[list[int]] = utils.replace_words_with_indices(
    sentences_in_words=sentences_in_words,
    word_index_dict=vocab,
    max_index=n_most_frequent_words
)

training_data: list[tuple[int, list[int]]] = utils.create_training_data(
    sentences_in_idxs=sentences_in_idxs,
    context_window_size=context_window_size
)

random.shuffle(training_data)

print("there are " + str(len(training_data)) + " training samples")


processed_data: dict[str, Union[int, int, list[tuple[int, list[int]]]]] = {
    "vocab_size": n_most_frequent_words,
    "context_window_size": context_window_size,
    "training_data": training_data
}

with open("./v2/data/processing/processed_data.pickle", "wb") as handle:
    pickle.dump(
        processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL
    )

vocab_data: dict[str, Union[int, dict[str, tuple[int, int]]]] = {
    "vocab_size": n_most_frequent_words,
    "vocab": vocab
}

with open("./v2/data/processing/vocab_data.pickle", "wb") as handle:
    pickle.dump(
        vocab_data, handle, protocol=pickle.HIGHEST_PROTOCOL
    )
