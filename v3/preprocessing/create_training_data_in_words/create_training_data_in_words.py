import v3.preprocessing.preprocessing_utils as preprocessing_utils


def create_training_data_in_words(
        sentences_in_words: list[list[str]],
        context_window_size: int
) -> list[tuple[str, list[str]]]:
    return preprocessing_utils.flatten([
        preprocessing_utils.create_training_data_in_words_for_sentence(
            sentence_in_words=sentence_in_words,
            context_window_size=context_window_size
        ) for sentence_in_words in sentences_in_words
    ])
