def create_idx_to_vocab_freq(
        word_to_idx_count_vocab: dict[str, tuple[int, int]]
) -> dict[int, float]:
    total_count: int = sum(
        [
            idx_and_count[1] for idx_and_count in word_to_idx_count_vocab.values()
        ]
    )

    return {
        idx_and_count[0]: idx_and_count[1] / total_count for idx_and_count in word_to_idx_count_vocab.values()
    }
