from .reduce_training_data_and_map_to_idxs import reduce_training_data_and_map_to_idxs


def runTest():
    training_data_in_words: list[tuple[str, list[str]]] = [
        ("hello", ["i", "am"]),
        ("i", ["hello", "am", "a"]),
        ("am", ["hello", "i", "a", "preprocessed"]),
        ("a", ["i", "am", "preprocessed", "sentence"]),
        ("preprocessed", ["am", "a", "sentence"]),
        ("sentence", ["a", "preprocessed"])
    ]

    word_to_idx_count_vocab: dict[str, tuple[int, int]] = {
        "i": (1, 7),
        "am": (0, 8),
        "a": (2, 5),
        "preprocessed": (3, 4)
    }

    output: list[tuple[int, list[int]]] = reduce_training_data_and_map_to_idxs(
        full_training_data_in_words=training_data_in_words,
        word_to_idx_count_vocab=word_to_idx_count_vocab,
    )

    assert output == [
        (1, [0, 2]),
        (0, [1, 2, 3]),
        (2, [1, 0, 3]),
        (3, [0, 2])
    ]
