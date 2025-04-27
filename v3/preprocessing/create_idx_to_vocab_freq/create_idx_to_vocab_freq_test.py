from .create_idx_to_vocab_freq import create_idx_to_vocab_freq


def runTest():
    word_to_idx_count_vocab: dict[str, tuple[int, int]] = {
        "man": (0, 16),
        "woman": (1, 8),
        "cat": (2, 4),
        "dog": (3, 2)
    }

    output: dict[int, float] = create_idx_to_vocab_freq(
        word_to_idx_count_vocab=word_to_idx_count_vocab
    )

    assert output == {
        0: 16 / 30,
        1: 8 / 30,
        2: 4 / 30,
        3: 2/30
    }
