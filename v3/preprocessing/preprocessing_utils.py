import re
import v2.utils.utils as utils


def split_into_sentences(text: str) -> list[str]:
    pattern = r"[.!?]\s+"
    return re.split(pattern, text)


def remove_punctuation(sentences: list[str]) -> list[str]:
    pattern = r"[^\w\s]"
    return [re.sub(pattern, "", sentence) for sentence in sentences]


def lower(sentences: list[str]) -> list[str]:
    return [sentence.lower() for sentence in sentences]


def split_into_words(sentences: list[str]) -> list[list[str]]:
    return [sentence.split() for sentence in sentences]


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


def create_training_data_in_words_for_sentence(
        sentence_in_words: list[str],
        context_window_size: int
) -> list[tuple[str, list[str]]]:
    return [
        (word, create_bag_of_words(
            word_idx_in_sentence=word_idx_in_sentence,
            sentence_in_words=sentence_in_words,
            context_window_size=context_window_size
        )) for word_idx_in_sentence, word in enumerate(sentence_in_words) if len(create_bag_of_words(
            word_idx_in_sentence=word_idx_in_sentence,
            sentence_in_words=sentence_in_words,
            context_window_size=context_window_size
        )) > 0
    ]


def create_training_data_in_words(
        sentences_in_words: list[list[str]],
        context_window_size: int
) -> list[tuple[str, list[str]]]:
    return utils.flatten([
        create_training_data_in_words_for_sentence(
            sentence_in_words=sentence_in_words,
            context_window_size=context_window_size
        ) for sentence_in_words in sentences_in_words
    ])


def create_init_word_to_idx_count_vocab(
        sentences_in_words: list[list[str]]
) -> dict[str, tuple[int, int]]:
    return {word: (-1, 0) for word in set(utils.flatten(sentences_in_words))}


def add_window_count_to_word_to_idx_count_vocab_via_words(
        word_to_idx_count_vocab: dict[str, tuple[int, int]],
        training_data_in_words: list[tuple[str, list[str]]]
) -> dict[str, tuple[int, int]]:
    windows: list[set[str]] = [
        set([sample[0]] + sample[1]) for sample in training_data_in_words
    ]

    for window in windows:
        for word in window:
            word_to_idx_count_vocab[word] = (
                word_to_idx_count_vocab[word][0], word_to_idx_count_vocab[word][1] + 1)

    return word_to_idx_count_vocab


def reduce_vocab_via_window_count(
        full_word_to_idx_count_vocab: dict[str, tuple[int, int]],
        reduced_vocab_size: int
) -> dict[str, tuple[int, int]]:
    vocab_sorted_by_window_count: dict[str, tuple[int, int]] = dict(
        sorted(full_word_to_idx_count_vocab.items(),
               key=lambda item: item[1][1], reverse=True)
    )

    return {
        v[0]: (v_idx, v[1][1]) for v_idx, v in enumerate(vocab_sorted_by_window_count.items()) if v_idx < reduced_vocab_size
    }


def reduce_context_window(
    context_window: list[str],
    word_to_idx_count_vocab: dict[str, tuple[int, int]],
) -> list[int]:
    return [word_to_idx_count_vocab[word][0] for word in context_window if word in word_to_idx_count_vocab]


def reduce_training_data(
        full_training_data_in_words: list[tuple[str, list[str]]],
        word_to_idx_count_vocab: dict[str, tuple[int, int]],
) -> list[tuple[int, list[int]]]:
    return [
        (word_to_idx_count_vocab[sample[0]][0], reduced_context_window) for sample in full_training_data_in_words if (
            sample[0] in word_to_idx_count_vocab
        ) and (
            reduced_context_window := reduce_context_window(
                context_window=sample[1],
                word_to_idx_count_vocab=word_to_idx_count_vocab
            )
        )
    ]


# todo for sebstemmer: test (already exists)
def create_idx_to_word_vocab(
        word_to_idx_count_vocab: dict[str, tuple[int, int]]
) -> dict[int, str]:
    return {
        word_idx_and_count[0]: word for word, word_idx_and_count in word_to_idx_count_vocab.items()
    }


# todo for sebstemmer: test
def add_window_count_to_vocab(
        word_to_idx_count_vocab: dict[str, tuple[int, int]],
        idx_to_word_vocab: dict[int, str],
        training_data: list[tuple[int, list[int]]]
) -> dict[str, tuple[int, int]]:
    windows: list[set[int]] = [
        set([sample[0]] + sample[1]) for sample in training_data
    ]

    for word in word_to_idx_count_vocab.keys():
        word_to_idx_count_vocab[word] = (word_to_idx_count_vocab[word][0], 0)

    for window in windows:
        for word_idx in window:
            word: str = idx_to_word_vocab[word_idx]
            word_to_idx_count_vocab[word] = (
                word_to_idx_count_vocab[word][0], word_to_idx_count_vocab[word][1] + 1)

    return word_to_idx_count_vocab


# todo for sebstemmer: test (already exists)
def create_word_to_idx_vocab(
        word_to_idx_count_vocab: dict[str, tuple[int, int]]
) -> dict[str, int]:
    return {
        word: word_idx_and_count[0] for word, word_idx_and_count in word_to_idx_count_vocab.items()
    }


class PreprocessedData:
    def __init__(
            self,
            context_window_size: int,
            training_data: list[tuple[int, list[int]]]
    ):
        self.context_window_size: int = context_window_size
        self.training_data: list[tuple[int, list[int]]] = training_data


class Vocab:
    def __init__(
            self,
            vocab_size: int,
            word_to_idx_vocab: dict[str, int],
            idx_to_word_vocab: dict[int, str],
            word_to_idx_count_vocab: dict[str, tuple[int, int]]
    ):
        self.vocab_size: int = vocab_size
        self.word_to_idx_vocab: dict[str, int] = word_to_idx_vocab
        self.idx_to_word_vocab: dict[int, str] = idx_to_word_vocab
        self.word_to_idx_count_vocab: dict[
            str,
            tuple[int, int]
        ] = word_to_idx_count_vocab
