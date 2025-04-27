from .reduce_context_window_and_map_to_idx import reduce_context_window_and_map_to_idx


def runTest():
    context_window: list[str] = ["am", "a", "sentence"]

    word_to_idx_count_vocab: dict[str, tuple[int, int]] = {
        "i": (1, 7),
        "am": (0, 8),
        "a": (2, 5),
        "preprocessed": (3, 4)
    }

    output: list[int] = reduce_context_window_and_map_to_idx(
        context_window=context_window,
        word_to_idx_count_vocab=word_to_idx_count_vocab
    )

    assert output == [0, 2]
