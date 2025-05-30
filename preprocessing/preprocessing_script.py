import random
import pickle
import preprocessing.preprocessing_utils as preprocessing_utils


vocab_size: int = 30000
context_window_size: int = 4
preprocessing_run_label: str = "vs_30_cw_4"


print("start preprocessing-run with label " + preprocessing_run_label + "...")

print("read raw text...")

raw_text: str = open(preprocessing_utils.path_to_data_folder +
                     "raw/AllCombined.txt", "r").read()

print("...raw text read")


print("split into sentences...")

sentences: list[str] = preprocessing_utils.split_into_sentences(raw_text)

print("...split into sentences")


print("remove punctuation...")

sentences: list[str] = preprocessing_utils.remove_punctuation(sentences)

print("...punctuation removed")


print("lower sentences...")

sentences: list[str] = preprocessing_utils.lower(sentences)

print("...sentences lowered")


print("split into words...")

sentences_in_words: list[list[str]] = preprocessing_utils.split_into_words(
    sentences
)

print("...split into words")


print("create full training-data-in-words...")

full_training_data_in_words: list[tuple[str, list[str]]] = preprocessing_utils.create_training_data_in_words(
    sentences_in_words=sentences_in_words,
    context_window_size=context_window_size
)

print("...full training-data-in-words created")


print("create full word-to-idx-count-vocab...")

full_word_to_idx_count_vocab: dict[str, tuple[int, int]] = preprocessing_utils.create_init_word_to_idx_count_vocab(
    sentences_in_words=sentences_in_words
)

print("...full word-to-idx-count-vocab created")


print("add window count to full vocab...")

full_vocab_with_window_count = preprocessing_utils.add_window_count_to_word_to_idx_count_vocab_via_words(
    word_to_idx_count_vocab=full_word_to_idx_count_vocab,
    training_data_in_words=full_training_data_in_words
)

print("...added window count to full vocab")


print("reduce vocab via window size...")

reduced_word_to_idx_count_vocab: dict[str, tuple[int, int]] = preprocessing_utils.reduce_word_to_idx_count_vocab_via_window_count(
    full_word_to_idx_count_vocab=full_word_to_idx_count_vocab,
    reduced_vocab_size=vocab_size
)

print("...reduced vocab via window size")


print("reduce training-data and map to idxs...")

training_data: list[tuple[int, list[int]]] = preprocessing_utils.reduce_training_data_and_map_to_idxs(
    full_training_data_in_words=full_training_data_in_words,
    word_to_idx_count_vocab=reduced_word_to_idx_count_vocab
)

print("...reduced training-data and mapped to idxs")


print("create idx-to-word-vocab...")

idx_to_word_vocab: dict[int, str] = preprocessing_utils.create_idx_to_word_vocab(
    word_to_idx_count_vocab=reduced_word_to_idx_count_vocab
)

print("...idx-to-word-vocab created")


print("add window count to word-to-idx-count-vocab...")

word_to_idx_count_vocab: dict[str, tuple[int, int]] = preprocessing_utils.add_window_count_to_word_to_idx_count_vocab(
    word_to_idx_count_vocab=reduced_word_to_idx_count_vocab,
    idx_to_word_vocab=idx_to_word_vocab,
    training_data=training_data
)

print("...added window count to word-to-idx-count-vocab")


print("create word-to-idx-vocab...")

word_to_idx_vocab: dict[str, int] = preprocessing_utils.create_word_to_idx_vocab(
    word_to_idx_count_vocab=word_to_idx_count_vocab
)

print("...word-to-idx-vocab created")


print("create idx-to-vocab-freq...")

idx_to_vocab_freq: dict[int, float] = preprocessing_utils.create_idx_to_vocab_freq(
    word_to_idx_count_vocab=word_to_idx_count_vocab
)

print("...idx-to-vocab-freq created")


print("shuffle training-data...")

random.shuffle(training_data)

print("...shuffled training-data")


print("save preprocessed data...")

preprocessed_data: preprocessing_utils.PreprocessedData = preprocessing_utils.PreprocessedData(
    context_window_size=context_window_size,
    training_data=training_data
)

with open(preprocessing_utils.path_to_preprocessed_data(
    preprocessing_run_label=preprocessing_run_label
), "wb") as handle:
    pickle.dump(
        preprocessed_data, handle, protocol=pickle.HIGHEST_PROTOCOL
    )

print("...preprocessed data saved")


print("save vocab...")

vocab: preprocessing_utils.Vocab = preprocessing_utils.Vocab(
    vocab_size=vocab_size,
    word_to_idx_vocab=word_to_idx_vocab,
    idx_to_word_vocab=idx_to_word_vocab,
    word_to_idx_count_vocab=word_to_idx_count_vocab,
    idx_to_vocab_freq=idx_to_vocab_freq
)

with open(preprocessing_utils.path_to_vocab(
    preprocessing_run_label=preprocessing_run_label
), "wb") as handle:
    pickle.dump(
        vocab, handle, protocol=pickle.HIGHEST_PROTOCOL
    )

print("...vocab saved")
