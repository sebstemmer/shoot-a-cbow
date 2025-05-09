from typing import Callable


def subsample_idx(
        idx: int,
        subsampling_t: float,
        subsampling_pow: float,
        idx_to_vocab_freq: dict[int, float],
        generate_random_between_0_and_1: Callable[[], float]
) -> int:
    frequency: float = idx_to_vocab_freq[idx]

    term: float = (subsampling_t / frequency) ** subsampling_pow

    if term >= 1:
        return idx

    random_between_0_and_1: float = generate_random_between_0_and_1()

    if random_between_0_and_1 > term:
        return -1
    else:
        return idx
