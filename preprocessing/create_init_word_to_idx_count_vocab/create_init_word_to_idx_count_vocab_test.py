from .create_init_word_to_idx_count_vocab import create_init_word_to_idx_count_vocab


def runTest():
    sentences_in_words: list[list[str]] = [
        ["hello", "i", "am", "a", "preprocessed", "sentence"],
        ["i", "am", "sure"]
    ]

    output: dict[
        str, tuple[int, int]
    ] = create_init_word_to_idx_count_vocab(
        sentences_in_words=sentences_in_words
    )

    assert output == {
        "hello": (-1, 0),
        "i": (-1, 0),
        "am": (-1, 0),
        "a": (-1, 0),
        "preprocessed": (-1, 0),
        "sentence": (-1, 0),
        "sure": (-1, 0)
    }
