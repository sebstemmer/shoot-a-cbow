from .create_training_data_in_words_for_sentence import create_training_data_in_words_for_sentence


def runTest():
    sentence_in_words: list[str] = [
        "hello", "i", "am", "a", "preprocessed", "sentence"
    ]

    output: list[
        tuple[str, list[str]]
    ] = create_training_data_in_words_for_sentence(
        sentence_in_words=sentence_in_words,
        context_window_size=2
    )

    assert output == [
        ("hello", ["i", "am"]),
        ("i", ["hello", "am", "a"]),
        ("am", ["hello", "i", "a", "preprocessed"]),
        ("a", ["i", "am", "preprocessed", "sentence"]),
        ("preprocessed", ["am", "a", "sentence"]),
        ("sentence", ["a", "preprocessed"])
    ]
