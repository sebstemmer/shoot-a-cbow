from .split_into_sentences import split_into_sentences


def runTest():
    text: str = "Hello, I am a sentence. Why are you splitting me?\n\n Are you crazy! Why are you, doing this to me?"

    output: list[str] = split_into_sentences(
        text=text
    )

    assert output == [
        "Hello, I am a sentence",
        "Why are you splitting me",
        "Are you crazy",
        "Why are you, doing this to me?"
    ]
