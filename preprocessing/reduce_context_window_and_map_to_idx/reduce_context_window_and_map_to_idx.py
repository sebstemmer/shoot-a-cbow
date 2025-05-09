def reduce_context_window_and_map_to_idx(
    context_window: list[str],
    word_to_idx_count_vocab: dict[str, tuple[int, int]],
) -> list[int]:
    return [
        word_to_idx_count_vocab[word][0] for word in context_window if word in word_to_idx_count_vocab
    ]
