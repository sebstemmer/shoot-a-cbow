from .create_bag_of_words import create_bag_of_words


def runTest():
    sentence_in_words: list[str] = [
        "hi", "wow", "dog", "cat", "fish", "wow"
    ]

    output_0: list[str] = create_bag_of_words(
        word_idx_in_sentence=1,
        sentence_in_words=sentence_in_words,
        context_window_size=2
    )

    assert output_0 == ["hi", "dog", "cat"]

    output_1 = create_bag_of_words(
        word_idx_in_sentence=4,
        sentence_in_words=sentence_in_words,
        context_window_size=2
    )

    assert output_1 == ["dog", "cat", "wow"]

    output_2 = create_bag_of_words(
        word_idx_in_sentence=3,
        sentence_in_words=sentence_in_words,
        context_window_size=1
    )

    assert output_2 == ["dog", "fish"]
