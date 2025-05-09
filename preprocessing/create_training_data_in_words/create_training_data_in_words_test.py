from .create_training_data_in_words import create_training_data_in_words


def runTest():
    sentences_in_words: list[list[str]] = [
        ["hello", "i", "am", "a", "preprocessed", "sentence"],
        ["are", "you", "sure"]
    ]

    output: list[
        tuple[str, list[str]]
    ] = create_training_data_in_words(
        sentences_in_words=sentences_in_words,
        context_window_size=2
    )

    assert output == [
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
