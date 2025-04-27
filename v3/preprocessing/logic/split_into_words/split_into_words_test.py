from .split_into_words import split_into_words


def runTest():
    sentences: list[str] = [
        "hello i am a sentence",
        "why are you splitting me",
    ]

    output: list[list[str]] = split_into_words(
        sentences=sentences
    )

    assert output == [
        ["hello", "i", "am", "a", "sentence"],
        ["why", "are", "you", "splitting", "me"]
    ]
