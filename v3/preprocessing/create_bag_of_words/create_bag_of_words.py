def create_bag_of_words(
        word_idx_in_sentence: int,
        sentence_in_words: list[str],
        context_window_size: int
) -> list[str]:
    return [
        sentence_in_words[idx_in_sentence] for idx_in_sentence in range(
            word_idx_in_sentence-context_window_size,
            word_idx_in_sentence+context_window_size + 1
        ) if (idx_in_sentence >= 0 and idx_in_sentence < len(sentence_in_words) and idx_in_sentence != word_idx_in_sentence)
    ]
