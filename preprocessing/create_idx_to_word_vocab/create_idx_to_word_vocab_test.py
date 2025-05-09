from .create_idx_to_word_vocab import create_idx_to_word_vocab


def runTest():
    word_to_idx_count_vocab: dict[str, tuple[int, int]] = {
        "some": (0, 121),
        "word": (1, 12),
        "is": (2, 3),
        "good": (3, 1),
    }

    output: dict[int, str] = create_idx_to_word_vocab(
        word_to_idx_count_vocab=word_to_idx_count_vocab
    )

    assert output == {
        0: "some",
        1: "word",
        2: "is",
        3: "good"
    }
