from typing import TypeVar

T = TypeVar('T')

path_to_data: str = "./v3/data/"


def flatten(listOfLists: list[list[T]]) -> list[T]:
    return [item for sublist in listOfLists for item in sublist]


def remove_elements_from_list(list: list[T], remove: list[T]) -> list[T]:
    remove_set: set[T] = set(remove)
    return [el for el in list if el not in remove_set]


def print_progress(array: list[T], idx: int, on: bool, acc: int = 1):
    if (not on):
        return

    tenthArraySize = round(len(array) / pow(10, acc))

    if (tenthArraySize < 1):
        raise Exception(tenthArraySize + " < 1")

    for n in range(1, pow(10, acc)):
        if (n * tenthArraySize == idx):
            print(str(round(n * (pow(10, -acc+2)), acc)) + " %")
            return
