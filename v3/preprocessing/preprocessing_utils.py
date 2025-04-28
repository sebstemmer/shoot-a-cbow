from .split_into_sentences.split_into_sentences import *
from .remove_punctuation.remove_punctuation import *
from .lower.lower import *
from .split_into_words.split_into_words import *
from .create_bag_of_words.create_bag_of_words import *
from .create_training_data_in_words_for_sentence.create_training_data_in_words_for_sentence import *
from .create_training_data_in_words.create_training_data_in_words import *
from .add_window_count_to_word_to_idx_count_vocab_via_words.add_window_count_to_word_to_idx_count_vocab_via_words import *
from .reduce_context_window_and_map_to_idx.reduce_context_window_and_map_to_idx import *
from .reduce_training_data_and_map_to_idxs.reduce_training_data_and_map_to_idxs import *
from .create_idx_to_word_vocab.create_idx_to_word_vocab import *
from .add_window_count_to_word_to_idx_count_vocab.add_window_count_to_word_to_idx_count_vocab import *
from .create_word_to_idx_vocab.create_word_to_idx_vocab import *
from .create_init_word_to_idx_count_vocab.create_init_word_to_idx_count_vocab import *
from .reduce_word_to_idx_count_vocab_via_window_count.reduce_word_to_idx_count_vocab_via_window_count import *
from .subsample_idx.subsample_idx import *
from .subsample_training_data.subsample_training_data import *
from .create_idx_to_vocab_freq.create_idx_to_vocab_freq import *
from .flatten.flatten import *
from .create_sorted_idx_to_log_count_vocab.create_sorted_idx_to_log_count_vocab import *


path_to_data_folder: str = "./v3/data/"


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
            word_to_idx_count_vocab: dict[str, tuple[int, int]],
            idx_to_vocab_freq: dict[int, float]
    ):
        self.vocab_size: int = vocab_size
        self.word_to_idx_vocab: dict[str, int] = word_to_idx_vocab
        self.idx_to_word_vocab: dict[int, str] = idx_to_word_vocab
        self.word_to_idx_count_vocab: dict[
            str,
            tuple[int, int]
        ] = word_to_idx_count_vocab
        self.idx_to_vocab_freq: dict[int, float] = idx_to_vocab_freq
