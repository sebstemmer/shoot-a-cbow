import v3.preprocessing.preprocessing_logic as preprocessing_logic


def create_init_word_to_idx_count_vocab(
        sentences_in_words: list[list[str]]
) -> dict[str, tuple[int, int]]:
    return {word: (-1, 0) for word in set(preprocessing_logic.flatten(sentences_in_words))}
