import v2.data_processing.data_processing_utils as utils

# todo for sebstemmer: überall typen + richtige benennung

# split_into_sentences

input = "Hello, I am a sentence. Why are you splitting me?\n\n Are you crazy! Why are you, doing this to me?"

split_into_sentences_output = utils.split_into_sentences(
    input
)

assert split_into_sentences_output == [
    "Hello, I am a sentence",
    "Why are you splitting me",
    "Are you crazy",
    "Why are you, doing this to me?"
]

# remove_punctuation

input = [
    "Hello, I am a. sentence1 12",
    "Why are you! splitting me.",
]

remove_punctuation_output = utils.remove_punctuation(input)

assert remove_punctuation_output == [
    "Hello I am a sentence1 12",
    "Why are you splitting me",
]

# lower

input = [
    "Hello I am a sentence",
    "Why are you splitting me",
]

output = utils.lower(input)

assert output == [
    "hello i am a sentence",
    "why are you splitting me",
]


# split into words

input = [
    "hello i am a sentence",
    "why are you splitting me",
]

output = utils.split_into_words(input)

assert output == [
    ["hello", "i", "am", "a", "sentence"],
    ["why", "are", "you", "splitting", "me"]
]


# count_words_and_create_indices

input = [
    ["hello", "me", "am", "a", "sentence", "me"],
    ["why", "are", "you", "splitting", "me", "into", "more", "sentence"]
]

output = utils.count_words_and_create_indices(input)

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

output = utils.replace_words_with_indices(
    sentences_in_words=input,
    word_index_dict=vocab,
    max_index=9
)

assert output == [[2, 0, 3, 4, 1, 0], [5, 6, 7, 8, 0, -1, -1, 1]]


# create_bag_of_words

input = [3, 1, 4, 5, 2, 1, -1]

output = utils.create_bag_of_words(
    word_idx=1,
    sentence_in_idxs=input,
    context_window_size=2
)

assert output == [3, 4, 5]

output = utils.create_bag_of_words(
    word_idx=4,
    sentence_in_idxs=input,
    context_window_size=2
)

assert output == [4, 5, 1]

output = utils.create_bag_of_words(
    word_idx=3,
    sentence_in_idxs=input,
    context_window_size=3
)

assert output == [3, 1, 4, 2, 1]


# create_training_data_for_sentence

input = [3, 1, 4, -1, 2, 1]

output = utils.create_training_data_for_sentence(
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

output = utils.create_training_data(
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
]
