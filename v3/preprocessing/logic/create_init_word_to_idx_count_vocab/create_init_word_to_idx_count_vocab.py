import v3.utils.utils as utils  # todo for sebstemmer


def create_init_word_to_idx_count_vocab(
        sentences_in_words: list[list[str]]
) -> dict[str, tuple[int, int]]:
    return {word: (-1, 0) for word in set(utils.flatten(sentences_in_words))}
