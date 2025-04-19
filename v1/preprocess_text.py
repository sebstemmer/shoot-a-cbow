import string
import nltk
from nltk.corpus import stopwords
import util
import re
import pickle

min_word_count: int = 100
sentence_ending: string = "<sen>"

nltk.download('stopwords')
nltk.download('punkt_tab')
stops = set(stopwords.words('english'))
# print(stops)

# read file

file = open("/Users/sebstemmer/tmp/250316/AllCombined.txt", "r")
raw_text: string = file.read()

# preprocess text

print(len(raw_text))

removed_punctuation_and_numbers: string = util.remove_punctuation_except_sentence_endings_and_numbers(
    raw_text
)
replaced_sentence_endings: string = util.replace_sentence_endings(
    removed_punctuation_and_numbers, " " + sentence_ending + " "
)
removed_one_char_words: string = util.remove_one_char_words(
    replaced_sentence_endings)
removed_multi_spaces_and_new_lines: string = util.remove_multi_spaces_and_new_lines(
    removed_one_char_words
)
lowered: string = removed_multi_spaces_and_new_lines.lower()


counts = util.count_words(lowered)

in_frequent_words = [k for k, v in counts.items() if v < min_word_count]

words_for_removal = stops.union(
    set(in_frequent_words)
)

lowered_words = lowered.split()

after_words_removed = []
for idx, word in enumerate(lowered_words):
    util.print_progress(lowered_words, idx, True, 1)
    if (word not in words_for_removal):
        after_words_removed.append(word)

text_after_words_removed = " ".join(after_words_removed)

split_in_sentences = re.split(
    re.compile(
        sentence_ending
    ),
    text_after_words_removed
)

split_in_sentences_and_words = [
    sentence.split() for sentence in split_in_sentences
]

with open('preprocessed_text.pickle', 'wb') as handle:
    pickle.dump(
        split_in_sentences_and_words, handle, protocol=pickle.HIGHEST_PROTOCOL
    )
