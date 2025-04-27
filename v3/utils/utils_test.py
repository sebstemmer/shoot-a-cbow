import v2.utils.utils as utils


# flatten

flatten_input: list[list[str]] = [
    ["hello", "i", "am", "a", "sentence"],
    ["why", "are", "you", "splitting", "me"]
]

flatten_output: list[str] = utils.flatten(
    flatten_input
)

assert flatten_output == [
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


# remove_elements_from_list

remove_elements_from_list_input: list[str] = [
    "hello", "i", "am", "a", "sentence"
]

remove_elements_from_list_remove: list[str] = [
    "hello", "a"
]

remove_elements_from_list_output: list[str] = utils.remove_elements_from_list(
    remove_elements_from_list_input,
    remove_elements_from_list_remove
)

assert remove_elements_from_list_output == [
    "i",
    "am",
    "sentence"
]
