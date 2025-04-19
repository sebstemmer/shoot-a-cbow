import math
from typing import Callable
import random


# look at the columns of the embeddings

def create_vocab_frequencies(vocab: dict[str, tuple[int, int]]) -> dict[int, float]:
    total_count: int = sum([countAndIdx[0] for countAndIdx in vocab.values()])

    return {
        countAndIdx[1]: countAndIdx[0] / total_count for countAndIdx in vocab.values()
    }


def subsample_training_data(
    training_data: list[tuple[int, list[int]]],
    vocab_frequencies: dict[int, float],
    subsampling_t: float,
    generateRandomBetween0And1:  Callable[
        [],
        float
    ] = lambda: random.random()
) -> list[tuple[int, list[int]]]:
    subsampled_training_data: list[tuple[int, list[int]]] = []
    for training_sample in training_data:
        target_idx = subsample_idx(
            idx=training_sample[0],
            subsampling_t=subsampling_t,
            vocab_frequencies=vocab_frequencies,
            generateRandomBetween0And1=generateRandomBetween0And1
        )

        if target_idx != 0:
            filtered_context = list(
                filter(
                    lambda context_idx: subsample_idx(
                        idx=context_idx,
                        subsampling_t=subsampling_t,
                        vocab_frequencies=vocab_frequencies,
                        generateRandomBetween0And1=generateRandomBetween0And1
                    ) != 0, training_sample[1]
                )
            )

            if len(filtered_context) > 0:
                subsampled_training_data.append(
                    (
                        target_idx,
                        filtered_context
                    )
                )

    return subsampled_training_data


def subsample_idx(
        idx: int,
        subsampling_t: float,
        vocab_frequencies: dict[int, float],
        generateRandomBetween0And1: Callable[[], float]
) -> int:
    frequency: float = vocab_frequencies[idx]

    term: float = math.sqrt(subsampling_t / frequency)

    if term >= 1:
        return idx

    randomBetween0And1: float = generateRandomBetween0And1()

    if randomBetween0And1 > term:
        return 0
    else:
        return idx
