from .remove_punctuation import remove_punctuation


def runTest():
    sentences: list[str] = [
        "Hello, I am a. sentence1 12",
        "Why are you! splitting me.",
    ]

    output: list[str] = remove_punctuation(
        sentences=sentences
    )

    assert output == [
        "Hello I am a sentence1 12",
        "Why are you splitting me",
    ]
