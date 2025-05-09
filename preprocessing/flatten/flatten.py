from typing import TypeVar

T = TypeVar('T')


def flatten(list_of_lists: list[list[T]]) -> list[T]:
    return [item for sublist in list_of_lists for item in sublist]
