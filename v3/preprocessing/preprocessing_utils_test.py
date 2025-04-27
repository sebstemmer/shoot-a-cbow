import v3.preprocessing.preprocessing_utils as preprocessing_utils


# split_into_sentences

split_into_sentences_text: str = "Hello, I am a sentence. Why are you splitting me?\n\n Are you crazy! Why are you, doing this to me?"

split_into_sentences_output = preprocessing_utils.split_into_sentences(
    text=split_into_sentences_text
)

assert split_into_sentences_output == [
    "Hello, I am a sentence",
    "Why are you splitting me",
    "Are you crazy",
    "Why are you, doing this to me?"
]


# remove_punctuation

remove_punctuation_sentences: list[str] = [
    "Hello, I am a. sentence1 12",
    "Why are you! splitting me.",
]

remove_punctuation_output: list[str] = preprocessing_utils.remove_punctuation(
    sentences=remove_punctuation_sentences)

assert remove_punctuation_output == [
    "Hello I am a sentence1 12",
    "Why are you splitting me",
]


# lower

lower_sentences: list[str] = [
    "Hello I am a sentence",
    "Why are you splitting me",
]

lower_output: list[str] = preprocessing_utils.lower(
    sentences=lower_sentences
)

assert lower_output == [
    "hello i am a sentence",
    "why are you splitting me",
]


# split into words

split_into_words_sentences: list[str] = [
    "hello i am a sentence",
    "why are you splitting me",
]

split_into_words_output: list[list[str]] = preprocessing_utils.split_into_words(
    sentences=split_into_words_sentences
)

assert split_into_words_output == [
    ["hello", "i", "am", "a", "sentence"],
    ["why", "are", "you", "splitting", "me"]
]


# create_bag_of_words

create_bag_of_words_sentence_in_words: list[str] = [
    "hi", "wow", "dog", "cat", "fish", "wow"
]

create_bag_of_words_output_0: list[str] = preprocessing_utils.create_bag_of_words(
    word_idx_in_sentence=1,
    sentence_in_words=create_bag_of_words_sentence_in_words,
    context_window_size=2
)

assert create_bag_of_words_output_0 == ["hi", "dog", "cat"]

create_bag_of_words_output_1 = preprocessing_utils.create_bag_of_words(
    word_idx_in_sentence=4,
    sentence_in_words=create_bag_of_words_sentence_in_words,
    context_window_size=2
)

assert create_bag_of_words_output_1 == ["dog", "cat", "wow"]

create_bag_of_words_output_2 = preprocessing_utils.create_bag_of_words(
    word_idx_in_sentence=3,
    sentence_in_words=create_bag_of_words_sentence_in_words,
    context_window_size=1
)

assert create_bag_of_words_output_2 == ["dog", "fish"]


# create_training_data_in_words_for_sentence

create_training_data_in_words_for_sentence_sentence_in_words: list[str] = [
    "hello", "i", "am", "a", "preprocessed", "sentence"
]

create_training_data_in_words_for_sentence_output: list[
    tuple[str, list[str]]
] = preprocessing_utils.create_training_data_in_words_for_sentence(
    sentence_in_words=create_training_data_in_words_for_sentence_sentence_in_words,
    context_window_size=2
)

assert create_training_data_in_words_for_sentence_output == [
    ("hello", ["i", "am"]),
    ("i", ["hello", "am", "a"]),
    ("am", ["hello", "i", "a", "preprocessed"]),
    ("a", ["i", "am", "preprocessed", "sentence"]),
    ("preprocessed", ["am", "a", "sentence"]),
    ("sentence", ["a", "preprocessed"])
]


# create_training_data_in_words

create_training_data_in_words_sentences_in_words: list[list[str]] = [
    ["hello", "i", "am", "a", "preprocessed", "sentence"],
    ["are", "you", "sure"]
]

create_training_data_in_words_output: list[
    tuple[str, list[str]]
] = preprocessing_utils.create_training_data_in_words(
    sentences_in_words=create_training_data_in_words_sentences_in_words,
    context_window_size=2
)

assert create_training_data_in_words_output == [
    ("hello", ["i", "am"]),
    ("i", ["hello", "am", "a"]),
    ("am", ["hello", "i", "a", "preprocessed"]),
    ("a", ["i", "am", "preprocessed", "sentence"]),
    ("preprocessed", ["am", "a", "sentence"]),
    ("sentence", ["a", "preprocessed"]),
    ("are", ["you", "sure"]),
    ("you", ["are", "sure"]),
    ("sure", ["are", "you"])
]


# create_init_word_to_idx_count_vocab

create_init_word_to_idx_count_vocab_sentences_in_words: list[list[str]] = [
    ["hello", "i", "am", "a", "preprocessed", "sentence"],
    ["i", "am", "sure"]
]

create_init_word_to_idx_count_vocab_output: dict[
    str, tuple[int, int]
] = preprocessing_utils.create_init_word_to_idx_count_vocab(
    sentences_in_words=create_init_word_to_idx_count_vocab_sentences_in_words
)

assert create_init_word_to_idx_count_vocab_output == {
    "hello": (-1, 0),
    "i": (-1, 0),
    "am": (-1, 0),
    "a": (-1, 0),
    "preprocessed": (-1, 0),
    "sentence": (-1, 0),
    "sure": (-1, 0)
}


# add_window_count_to_word_to_idx_count_vocab_via_words

add_window_count_to_word_to_idx_count_vocab_via_words_word_to_idx_count_vocab: dict[
    str, tuple[int, int]
] = {
    "hello": (-1, 0),
    "i": (-1, 0),
    "am": (-1, 0),
    "a": (-1, 0),
    "preprocessed": (-1, 0),
    "sentence": (-1, 0),
    "sure": (-1, 0)
}

add_window_count_to_word_to_idx_count_vocab_via_words_training_data_in_words: list[tuple[str, list[str]]] = [
    ("hello", ["i", "am"]),
    ("i", ["hello", "am", "a"]),
    ("am", ["hello", "i", "a", "preprocessed"]),
    ("a", ["i", "am", "preprocessed", "sentence"]),
    ("preprocessed", ["am", "a", "sentence"]),
    ("sentence", ["a", "preprocessed"]),
    ("i", ["am", "sure"]),
    ("am", ["i", "sure"]),
    ("sure", ["i", "am"])
]

preprocessing_utils.add_window_count_to_word_to_idx_count_vocab_via_words(
    word_to_idx_count_vocab=add_window_count_to_word_to_idx_count_vocab_via_words_word_to_idx_count_vocab,
    training_data_in_words=add_window_count_to_word_to_idx_count_vocab_via_words_training_data_in_words
)

""" # count_words_and_create_indices

input = [
    ["hello", "me", "am", "a", "sentence", "me"],
    ["why", "are", "you", "splitting", "me", "into", "more", "sentence"]
]

output = preprocessing_utils.count_words_and_create_indices(input)

assert output == {"me": (3, 0),
                  "sentence": (2, 1),
                  "hello": (1, 2),
                  "am": (1, 3),
                  "a": (1, 4),
                  "why": (1, 5),
                  "are": (1, 6),
                  "you": (1, 7),
                  "splitting": (1, 8),
                  "into": (1, 9),
                  "more": (1, 10)
                  }


# replace_words_with_indices

input = [
    ["hello", "me", "am", "a", "sentence", "me"],
    ["why", "are", "you", "splitting", "me", "into", "more", "sentence"]
]

vocab = {"me": (3, 0),
         "sentence": (2, 1),
         "hello": (1, 2),
         "am": (1, 3),
         "a": (1, 4),
         "why": (1, 5),
         "are": (1, 6),
         "you": (1, 7),
         "splitting": (1, 8),
         "into": (1, 9),
         "more": (1, 10)
         }

output = preprocessing_utils.replace_words_with_indices(
    sentences_in_words=input,
    word_index_dict=vocab,
    max_index=9
)

assert output == [[2, 0, 3, 4, 1, 0], [5, 6, 7, 8, 0, -1, -1, 1]]


# create_bag_of_words

input = [3, 1, 4, 5, 2, 1, -1]

output = preprocessing_utils.create_bag_of_words(
    word_idx=1,
    sentence_in_idxs=input,
    context_window_size=2
)

assert output == [3, 4, 5]

output = preprocessing_utils.create_bag_of_words(
    word_idx=4,
    sentence_in_idxs=input,
    context_window_size=2
)

assert output == [4, 5, 1]

output = preprocessing_utils.create_bag_of_words(
    word_idx=3,
    sentence_in_idxs=input,
    context_window_size=3
)

assert output == [3, 1, 4, 2, 1]


# create_training_data_for_sentence

input = [3, 1, 4, -1, 2, 1]

output = preprocessing_utils.create_training_data_in_words_for_sentence(
    sentence_in_idxs=input,
    context_window_size=2
)

assert output == [
    (3, [1, 4]),
    (1, [3, 4]),
    (4, [3, 1, 2]),
    (2, [4, 1]),
    (1, [2])
]


# create_training_data

input = [[3, 1, 4, -1, 2, 1], [6, 7, 8, 9, 1, -1, -1, 2]]

output = preprocessing_utils.create_training_data_in_words(
    sentences_in_idxs=input,
    context_window_size=2
)

assert output == [
    (3, [1, 4]),
    (1, [3, 4]),
    (4, [3, 1, 2]),
    (2, [4, 1]),
    (1, [2]),
    (6, [7, 8]),
    (7, [6, 8, 9]),
    (8, [6, 7, 9, 1]),
    (9, [7, 8, 1]),
    (1, [8, 9])
] """
