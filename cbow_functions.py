import pickle


def create_vocab(input: list[list[str]]):
    distinct_words = list(
        set(
            [word for sentence in input for word in sentence]
        )
    )
    return {word: idx for idx, word in enumerate(distinct_words)}


def get_context(word_idx: int, sentence: list[int], context_window_size: int) -> list[int]:
    return [
        sentence[idx] for idx in range(
            word_idx-context_window_size,
            word_idx+context_window_size + 1
        ) if (idx >= 0 and idx < len(sentence) and idx != word_idx)
    ]


def create_training_data_for_sentence(sentence: list[int], context_window_size: int):
    result = []
    for word_idx, word in enumerate(sentence):
        context = get_context(word_idx, sentence, context_window_size)
        if len(context) > 0:
            result.append((word, context))

    return result

    # return [(word, get_context(word_idx, sentence, context_window_size)) for word_idx, word in enumerate(sentence)]


def create_training_data(sentences: list[list[int]], context_window_size: int):
    return flatten([create_training_data_for_sentence(sentence, context_window_size) for sentence in sentences])


def flatten(input):
    return [item for sublist in input for item in sublist]
