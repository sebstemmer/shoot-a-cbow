import string
import nltk
import collections
import re

stops = set(nltk.corpus.stopwords.words('english'))


def remove_punctuation_except_sentence_endings_and_numbers(input: string) -> string:
    pattern = r"[\!\?\.](?!\s)|[\"#$%&\'()*+,\-/:;<=>@[\]^_`{|}–]|\b\w*\d\w*\b"
    return re.sub(pattern, " ", input)


def replace_sentence_endings(input: string, replacement: string) -> string:
    pattern = r"(\.|\?|\!)\s"
    return re.sub(pattern, replacement, input)


def remove_one_char_words(input: string) -> string:
    pattern = r"\b[a-zA-Z]\b"
    return re.sub(pattern, "", input)


def remove_multi_spaces_and_new_lines(input: string) -> string:
    pattern = r"\s+"
    replacement = " "
    return re.sub(pattern, replacement, input)


def count_words(text) -> string:
    words = split_into_words(text)
    return collections.Counter(words)


def print_progress(array, idx: int, on: bool, acc=1):
    if (not on):
        return

    tenthArraySize = round(len(array) / pow(10, acc))

    if (tenthArraySize < 1):
        raise Exception(tenthArraySize + " < 1")

    for n in range(1, pow(10, acc)):
        if (n * tenthArraySize == idx):
            print(str(round(n * (pow(10, -acc+2)), acc)) + " %")
            return


def split_into_sentences(input: string, endingsRegex):
    sentences = re.split(endingsRegex, input)
    return sentences


# todo for sebstemmer: here old


def remove_punctuation(sentence):
    return ''.join([char for char in sentence if char not in string.punctuation])


def remove_multi_spaces_and_new_lines(input: string):
    pattern = r"\s+"
    replacement = " "
    return re.sub(pattern, replacement, input)


def split_into_words(input):
    return input.split()


def remove_words(text, remove_words):
    remove_words_as_set = set(remove_words)
    pattern = r'\b(?:' + '|'.join(map(re.escape,
                                      remove_words_as_set)) + r')\b(?:\s+|$)'
    return re.sub(pattern, '', text).strip()


def remove_stops(text):
    return remove_words(text, stops)


def preprocess_text(text):
    # removed_new_lines = remove_new_lines(text)
    lowered = text.lower()
    removed_stops = remove_stops(lowered)
    return removed_stops
