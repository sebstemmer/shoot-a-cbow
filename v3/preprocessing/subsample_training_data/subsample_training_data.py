from typing import Callable
import random
import v3.preprocessing.preprocessing_utils as preprocessing_utils


def subsample_training_data(
    training_data: list[tuple[int, list[int]]],
    idx_to_vocab_freq: dict[int, float],
    subsampling_t: float,
    subsampling_pow: float,
    generate_random_between_0_and_1:  Callable[
        [],
        float
    ] = lambda: random.random()
) -> list[tuple[int, list[int]]]:
    subsampled_training_data: list[tuple[int, list[int]]] = []

    for training_sample in training_data:
        target_idx = preprocessing_utils.subsample_idx(
            idx=training_sample[0],
            subsampling_t=subsampling_t,
            subsampling_pow=subsampling_pow,
            idx_to_vocab_freq=idx_to_vocab_freq,
            generate_random_between_0_and_1=generate_random_between_0_and_1
        )

        if target_idx != -1:
            filtered_context = list(
                filter(
                    lambda context_idx: preprocessing_utils.subsample_idx(
                        idx=context_idx,
                        subsampling_t=subsampling_t,
                        subsampling_pow=subsampling_pow,
                        idx_to_vocab_freq=idx_to_vocab_freq,
                        generate_random_between_0_and_1=generate_random_between_0_and_1
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
