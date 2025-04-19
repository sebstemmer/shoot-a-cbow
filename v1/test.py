import util


# remove_punctuation_except_sentence_endings_and_numbers

input = "Hi I am, a 21 sentence 42. with, some234some some punctuation? Maybe there is!a exclamation 2 mark! Why?do you think.that?"

output = util.remove_punctuation_except_sentence_endings_and_numbers(
    input)

assert output == "Hi I am  a   sentence  . with    some punctuation? Maybe there is a exclamation   mark! Why do you think that "


# replace_sentence_endings

input = "I am a sentence. I am another sentence! A question? Visit sebstemmer.com!"

output = util.replace_sentence_endings(input, " <S> ")

assert output == "I am a sentence <S> I am another sentence <S> A question <S> Visit sebstemmer.com!"


# replace_sentence_endings

input = "I am a sentence <sen> I am another sentence <sen> A question <sen> Visit sebstemmer <sen>"

output = util.remove_one_char_words(input)

assert output == " am  sentence <sen>  am another sentence <sen>  question <sen> Visit sebstemmer <sen>"


# remove_multi_spaces_and_new_lines

input = "\na\n\nb some  word.  I  am an example.  Sentence\n is good. "

removed = util.remove_multi_spaces_and_new_lines(
    input
)
print(removed)

assert removed == " a b some word. I am an example. Sentence is good. "


# count words

count_words = "hi I am an example sentence hi please count me and I am happy"
counted = util.count_words(count_words)

assert counted == {'hi': 2, 'I': 2, 'am': 2, 'an': 1, 'example': 1,
                   'sentence': 1, 'please': 1, 'count': 1, 'me': 1, 'and': 1, 'happy': 1}


# print_progress

a = range(0, 1000)
b = [chr(ord('a') + (el % 26)) for el in a]

for (idx, el) in enumerate(b):
    util.print_progress(b, idx, True, 2)


# todo for sebstemmer: here old


# remove_punctuation

sentence_with_punctuation = "hi I am, a sentence. with, some punctuation?"

removed_punctuation = util.remove_punctuation(sentence_with_punctuation)

assert removed_punctuation == "hi I am a sentence with some punctuation"


# split_into_words

sentence_with_words = " a green  house is red and blue"

splitted = util.split_into_words(sentence_with_words)

assert splitted == ["a", "green", "house", "is", "red", "and", "blue"]


# remove_words

text = "a green house is red and blue"

text_without_removed_words = util.remove_words(text, ["green", "is", "and"])

assert text_without_removed_words == "a house red blue"


# remove_stops

sentence_with_stops = "a green house is red and blue"

removed_stops = util.remove_stops(sentence_with_stops)

assert removed_stops == "green house red blue"


# preprocess_text

text = "In Europe, after the Middle Ages, there was a \"Renaissance\" which means \"rebirth\". People rediscovered science and artists were allowed to paint subjects other than religious subjects. People like Michelangelo and Leonardo da Vinci still painted religious pictures, but they also now could paint mythological pictures too. These artists also invented perspective where things in the distance look smaller in the picture. This was new because in the Middle Ages people would paint all the figures close up and just overlapping each other. These artists used nudity regularly in their art."
preprocessed_text = util.preprocess_text(text)

assert preprocessed_text == "europe, middle ages, \"renaissance\" means \"rebirth\". people rediscovered science artists allowed paint subjects religious subjects. people like michelangelo leonardo da vinci still painted religious pictures, also could paint mythological pictures too. artists also invented perspective things distance look smaller picture. new middle ages people would paint figures close overlapping other. artists used nudity regularly art."


# split into sentences

text = "hey i am some sentence. i am another sentence! i contain a comma, but i love being a sentence?"
text_split_in_sentences = util.split_into_sentences(text)

assert text_split_in_sentences == [
    "hey i am some sentence.",
    "i am another sentence!",
    "i contain a comma, but i love being a sentence?"
]
