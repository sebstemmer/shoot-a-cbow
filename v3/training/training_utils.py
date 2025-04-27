from typing import Callable
import random


# look at the columns of the embeddings

def create_vocab_frequencies(word_to_idx_count_vocab: dict[str, tuple[int, int]]) -> dict[int, float]:
    total_count: int = sum([idx_and_count[1]
                           for idx_and_count in word_to_idx_count_vocab.values()])

    return {
        idx_and_count[0]: idx_and_count[1] / total_count for idx_and_count in word_to_idx_count_vocab.values()
    }


def subsample_training_data(
    training_data: list[tuple[int, list[int]]],
    vocab_frequencies: dict[int, float],
    subsampling_t: float,
    subsampling_pow: float,
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
            subsampling_pow=subsampling_pow,
            vocab_frequencies=vocab_frequencies,
            generateRandomBetween0And1=generateRandomBetween0And1
        )

        if target_idx != -1:
            filtered_context = list(
                filter(
                    lambda context_idx: subsample_idx(
                        idx=context_idx,
                        subsampling_t=subsampling_t,
                        subsampling_pow=subsampling_pow,
                        vocab_frequencies=vocab_frequencies,
                        generateRandomBetween0And1=generateRandomBetween0And1
                    ) != -1, training_sample[1]
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
        subsampling_pow: float,
        vocab_frequencies: dict[int, float],
        generateRandomBetween0And1: Callable[[], float]
) -> int:
    frequency: float = vocab_frequencies[idx]

    term: float = (subsampling_t / frequency) ** subsampling_pow

    if term >= 1:
        return idx

    randomBetween0And1: float = generateRandomBetween0And1()

    if randomBetween0And1 > term:
        return -1
    else:
        return idx
