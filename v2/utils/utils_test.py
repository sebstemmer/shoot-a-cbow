import v2.utils.utils as utils


# flatten

input = [
    ["hello", "i", "am", "a", "sentence"],
    ["why", "are", "you", "splitting", "me"]
]

output = utils.flatten(
    input
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
