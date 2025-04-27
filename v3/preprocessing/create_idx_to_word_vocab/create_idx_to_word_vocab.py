def create_idx_to_word_vocab(
        word_to_idx_count_vocab: dict[str, tuple[int, int]]
) -> dict[int, str]:
    return {
        word_idx_and_count[0]: word for word, word_idx_and_count in word_to_idx_count_vocab.items()
    }
