def add_window_count_to_word_to_idx_count_vocab(
        word_to_idx_count_vocab: dict[str, tuple[int, int]],
        idx_to_word_vocab: dict[int, str],
        training_data: list[tuple[int, list[int]]]
) -> dict[str, tuple[int, int]]:
    windows: list[set[int]] = [
        set([sample[0]] + sample[1]) for sample in training_data
    ]

    for word in word_to_idx_count_vocab.keys():
        word_to_idx_count_vocab[word] = (word_to_idx_count_vocab[word][0], 0)

    for window in windows:
        for word_idx in window:
            word: str = idx_to_word_vocab[word_idx]
            word_to_idx_count_vocab[word] = (
                word_to_idx_count_vocab[word][0], word_to_idx_count_vocab[word][1] + 1)

    return word_to_idx_count_vocab
