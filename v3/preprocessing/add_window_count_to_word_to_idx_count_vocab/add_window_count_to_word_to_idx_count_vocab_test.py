from .add_window_count_to_word_to_idx_count_vocab import add_window_count_to_word_to_idx_count_vocab


def runTest():
    training_data: list[tuple[int, list[int]]] = [
        (1, [0, 2]),
        (0, [2, 3]),
        (2, [1, 0, 3]),
        (3, [0, 2])
    ]

    word_to_idx_count_vocab: dict[str, tuple[int, int]] = {
        "am": (0, 8),
        "i": (1, 7),
        "a": (2, 5),
        "preprocessed": (3, 4)
    }

    idx_to_word_vocab: dict[int, str] = {
        0: "am",
        1: "i",
        2: "a",
        3: "preprocessed"
    }

    output: dict[str, tuple[int, int]] = add_window_count_to_word_to_idx_count_vocab(
        word_to_idx_count_vocab=word_to_idx_count_vocab,
        idx_to_word_vocab=idx_to_word_vocab,
        training_data=training_data
    )

    assert output == {
        "am": (0, 4),
        "i": (1, 2),
        "a": (2, 4),
        "preprocessed": (3, 3)
    }
