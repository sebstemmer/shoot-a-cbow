from .create_sorted_idx_to_log_count_vocab import create_sorted_idx_to_log_count_vocab
import math


def runTest():
    word_to_idx_count_vocab: dict[str, tuple[int, int]] = {
        "hi": (0, 10),
        "i": (1, 13),
        "am": (2, 8),
        "a": (3, 9)
    }

    output: list[tuple[int, float]] = create_sorted_idx_to_log_count_vocab(
        word_to_idx_count_vocab=word_to_idx_count_vocab,
        log_safety=0.0
    )

    assert output == [
        (1, math.log(13.0)),
        (0, math.log(10.0)),
        (3, math.log(9.0)),
        (2, math.log(8.0))
    ]
