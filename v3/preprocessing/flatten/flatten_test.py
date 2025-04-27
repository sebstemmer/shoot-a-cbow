from .flatten import flatten


def runTest():
    list_of_lists: list[list[str]] = [
        ["hello", "i", "am", "a", "sentence"],
        ["why", "are", "you", "splitting", "me"]
    ]

    output: list[str] = flatten(
        list_of_lists=list_of_lists
    )

    assert output == [
        "hello",
        "i",
        "am",
        "a",
        "sentence",
        "why",
        "are",
        "you",
        "splitting",
        "me"
    ]
