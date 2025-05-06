def reduce_word_to_idx_count_vocab_via_window_count(
        full_word_to_idx_count_vocab: dict[str, tuple[int, int]],
        reduced_vocab_size: int
) -> dict[str, tuple[int, int]]:
    vocab_sorted_by_window_count: dict[str, tuple[int, int]] = dict(
        sorted(
            full_word_to_idx_count_vocab.items(),
            key=lambda item: item[1][1],
            reverse=True
        )
    )

    return {
        v[0]: (v_idx, v[1][1]) for v_idx, v in enumerate(vocab_sorted_by_window_count.items()) if v_idx < reduced_vocab_size
    }
