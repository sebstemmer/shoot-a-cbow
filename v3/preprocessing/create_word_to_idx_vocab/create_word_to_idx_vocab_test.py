from .create_word_to_idx_vocab import create_word_to_idx_vocab


def runTest():
    word_to_idx_count_vocab: dict[str, tuple[int, int]] = {
        "some": (0, 121),
        "word": (1, 12),
        "is": (2, 3),
        "good": (3, 1),
    }

    output: dict[str, int] = create_word_to_idx_vocab(
        word_to_idx_count_vocab=word_to_idx_count_vocab,
    )

    assert output == {
        "some": 0,
        "word": 1,
        "is": 2,
        "good": 3
    }
