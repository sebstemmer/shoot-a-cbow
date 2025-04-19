from typing import TypeVar

T = TypeVar('T')


def flatten(listOfLists: list[list[T]]) -> list[T]:
    return [item for sublist in listOfLists for item in sublist]
