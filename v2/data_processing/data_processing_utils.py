import re
import collections
import v2.utils.utils as utils
import torch


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


def count_words_and_create_indices(sentences_in_words: list[list[str]]) -> dict[str, tuple[int, int]]:
    counter = collections.Counter(utils.flatten(sentences_in_words))
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return {
        word: (count, idx) for idx, (word, count) in enumerate(sorted_items)
    }


def replace_words_with_indices(
        sentences_in_words: list[list[str]],
        word_index_dict: dict[str, tuple[int, int]],
        max_index: int
) -> list[list[int]]:
    return [
        [
            word_index_dict[word][1] if word_index_dict[word][1] < max_index else -1 for word in sentence_in_words
        ] for sentence_in_words in sentences_in_words
    ]


def create_bag_of_words(
        word_idx: int,
        sentence_in_idxs: list[int],
        context_window_size: int
) -> list[int]:
    return [
        sentence_in_idxs[idx] for idx in range(
            word_idx-context_window_size,
            word_idx+context_window_size + 1
        ) if (idx >= 0 and idx < len(sentence_in_idxs) and idx != word_idx and sentence_in_idxs[idx] != -1)
    ]

# todo for sebstemmer: richtige benennung von word_idx


def create_training_data_for_sentence(
        sentence_in_idxs: list[int],
        context_window_size: int
) -> list[tuple[int, list[int]]]:
    return [
        (word, create_bag_of_words(
            word_idx=word_idx,
            sentence_in_idxs=sentence_in_idxs,
            context_window_size=context_window_size
        )) for word_idx, word in enumerate(sentence_in_idxs) if len(create_bag_of_words(
            word_idx=word_idx,
            sentence_in_idxs=sentence_in_idxs,
            context_window_size=context_window_size
        )) > 0 and word != -1
    ]


def create_training_data(
        sentences_in_idxs: list[list[int]],
        context_window_size: int
) -> list[tuple[int, list[int]]]:
    return utils.flatten([
        create_training_data_for_sentence(
            sentence_in_idxs=sentence_in_idxs,
            context_window_size=context_window_size
        ) for sentence_in_idxs in sentences_in_idxs
    ])


def calc_entropy(
    training_data: list[tuple[int, list[int]]],
    vocab_size: int
):
    total_counts: torch.Tensor = torch.zeros(vocab_size)
    n_log_n: torch.Tensor = torch.zeros(vocab_size)

    for td_idx, training_sample in enumerate(training_data):
        print(td_idx)

        window: list[int] = [training_sample[0]] + training_sample[1]

        counts: torch.Tensor = torch.bincount(
            torch.tensor(window),
            minlength=vocab_size
        )

        n_log_n += counts * torch.log((counts+1e-10))
        total_counts += counts

    return torch.where(
        total_counts > 0,
        (-1.0 / total_counts) * (n_log_n - torch.log(total_counts) * total_counts),
        0.0
    )
