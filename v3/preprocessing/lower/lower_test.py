from .lower import lower


def runTest():
    sentences: list[str] = [
        "Hello I am a sentence",
        "Why are you splitting me",
    ]

    output: list[str] = lower(
        sentences=sentences
    )

    assert output == [
        "hello i am a sentence",
        "why are you splitting me",
    ]
