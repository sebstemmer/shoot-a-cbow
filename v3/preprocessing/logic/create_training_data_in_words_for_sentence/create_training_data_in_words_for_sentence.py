import v3.preprocessing.logic.create_bag_of_words.create_bag_of_words as create_bag_of_words


def create_training_data_in_words_for_sentence(
        sentence_in_words: list[str],
        context_window_size: int
) -> list[tuple[str, list[str]]]:
    return [
        (word, create_bag_of_words.create_bag_of_words(
            word_idx_in_sentence=word_idx_in_sentence,
            sentence_in_words=sentence_in_words,
            context_window_size=context_window_size
        )) for word_idx_in_sentence, word in enumerate(sentence_in_words) if len(create_bag_of_words.create_bag_of_words(
            word_idx_in_sentence=word_idx_in_sentence,
            sentence_in_words=sentence_in_words,
            context_window_size=context_window_size
        )) > 0
    ]
