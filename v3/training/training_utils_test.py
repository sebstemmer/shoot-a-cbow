import v2.training.training_utils as training_utils

# look at the columns of the embeddings

# create_vocab_frequencies

vocab: dict[str, tuple[int, int]] = {
    "man": (16, 0),
    "woman": (8, 1),
    "cat": (4, 2),
    "dog": (2, 3)
}

create_vocab_frequencies_output: dict[int, float] = training_utils.create_vocab_frequencies(
    vocab
)

assert create_vocab_frequencies_output == {
    0: 16 / 30,
    1: 8 / 30,
    2: 4 / 30,
    3: 2/30
}


# subsample_training_data

input = [
    (2, [0, 3]),
    (0, [2, 3]),
    (3, [2, 0, 1])
]

vocab_frequencies = {
    0: 0.1,
    1: 0.3,
    2: 0.2,
    3: 0.4
}

subsample_training_data_output: list[tuple[int, list[int]]] = training_utils.subsample_training_data(
    training_data=input,
    vocab_frequencies=vocab_frequencies,
    subsampling_t=1e-2,
    generateRandomBetween0And1=lambda: 0.2
)

assert subsample_training_data_output == [
    (2, [0]),
    (0, [2]),
]


# subsample_idx

vocab_frequencies: dict[int, float] = {
    1: 0.05,
    2: 0.3,
    3: 0.1,
    4: 0.07,
    5: 0.03,
    6: 0.05,
    7: 0.1,
    8: 0.15,
    9: 0.15
}

subsample_idx_output_0: int = training_utils.subsample_idx(
    idx=2,
    subsampling_t=1e-2,
    vocab_frequencies=vocab_frequencies,
    generateRandomBetween0And1=lambda: 0.1
)

assert subsample_idx_output_0 == 2


subsample_idx_output_1: int = training_utils.subsample_idx(
    idx=2,
    subsampling_t=1e-2,
    vocab_frequencies=vocab_frequencies,
    generateRandomBetween0And1=lambda: 0.2
)

assert subsample_idx_output_1 == -1
