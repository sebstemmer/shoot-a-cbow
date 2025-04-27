from .reduce_word_to_idx_count_vocab_via_window_count import reduce_word_to_idx_count_vocab_via_window_count


def runTest():
    word_to_idx_count_vocab: dict[str, tuple[int, int]] = {
        "hello": (-1, 3),
        "i": (-1, 7),
        "am": (-1, 8),
        "a": (-1, 5),
        "preprocessed": (-1, 4),
        "sentence": (-1, 3),
        "sure": (-1, 3)
    }

    output: dict[str, tuple[int, int]] = reduce_word_to_idx_count_vocab_via_window_count(
        full_word_to_idx_count_vocab=word_to_idx_count_vocab,
        reduced_vocab_size=4
    )

    assert output == {
        "i": (1, 7),
        "am": (0, 8),
        "a": (2, 5),
        "preprocessed": (3, 4)
    }
