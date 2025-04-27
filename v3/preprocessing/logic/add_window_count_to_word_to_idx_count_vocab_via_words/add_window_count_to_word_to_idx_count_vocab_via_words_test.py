from .add_window_count_to_word_to_idx_count_vocab_via_words import add_window_count_to_word_to_idx_count_vocab_via_words


def runTest():
    word_to_idx_count_vocab: dict[
        str, tuple[int, int]
    ] = {
        "hello": (-1, 0),
        "i": (-1, 0),
        "am": (-1, 0),
        "a": (-1, 0),
        "preprocessed": (-1, 0),
        "sentence": (-1, 0),
        "sure": (-1, 0)
    }

    training_data_in_words: list[tuple[str, list[str]]] = [
        ("hello", ["i", "am"]),
        ("i", ["hello", "am", "a"]),
        ("am", ["hello", "i", "a", "preprocessed"]),
        ("a", ["i", "am", "preprocessed", "sentence"]),
        ("preprocessed", ["am", "a", "sentence"]),
        ("sentence", ["a", "preprocessed"]),
        ("i", ["am", "sure"]),
        ("am", ["i", "sure"]),
        ("sure", ["i", "am"])
    ]

    output: dict[str, tuple[int, int]] = add_window_count_to_word_to_idx_count_vocab_via_words(
        word_to_idx_count_vocab=word_to_idx_count_vocab,
        training_data_in_words=training_data_in_words
    )

    assert output == {
        "hello": (-1, 3),
        "i": (-1, 7),
        "am": (-1, 8),
        "a": (-1, 5),
        "preprocessed": (-1, 4),
        "sentence": (-1, 3),
        "sure": (-1, 3)
    }
