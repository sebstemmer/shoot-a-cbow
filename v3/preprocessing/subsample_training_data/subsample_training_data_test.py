from .subsample_training_data import subsample_training_data


def runTest():
    training_data: list[tuple[int, list[int]]] = [
        (2, [0, 3]),
        (0, [2, 3]),
        (3, [2, 0, 1])
    ]

    idx_to_vocab_freq: dict[int, float] = {
        0: 0.1,
        1: 0.3,
        2: 0.2,
        3: 0.4
    }

    subsample_training_data_output: list[tuple[int, list[int]]] = subsample_training_data(
        training_data=training_data,
        idx_to_vocab_freq=idx_to_vocab_freq,
        subsampling_t=1e-2,
        subsampling_pow=0.5,
        generate_random_between_0_and_1=lambda: 0.2
    )

    assert subsample_training_data_output == [
        (2, [0]),
        (0, [2]),
    ]
