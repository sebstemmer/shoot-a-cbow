def add_window_count_to_word_to_idx_count_vocab_via_words(
        word_to_idx_count_vocab: dict[str, tuple[int, int]],
        training_data_in_words: list[tuple[str, list[str]]]
) -> dict[str, tuple[int, int]]:
    windows: list[set[str]] = [
        set([sample[0]] + sample[1]) for sample in training_data_in_words
    ]

    for window in windows:
        for word in window:
            word_to_idx_count_vocab[word] = (
                word_to_idx_count_vocab[word][0], word_to_idx_count_vocab[word][1] + 1)

    return word_to_idx_count_vocab
