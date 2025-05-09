def create_word_to_idx_vocab(
        word_to_idx_count_vocab: dict[str, tuple[int, int]]
) -> dict[str, int]:
    return {
        word: word_idx_and_count[0] for word, word_idx_and_count in word_to_idx_count_vocab.items()
    }
