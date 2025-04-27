from .subsample_idx import subsample_idx


def runTest():
    idx_to_vocab_freq: dict[int, float] = {
        1: 0.05,
        2: 0.3,
        3: 0.1,
        4: 0.07,
        5: 0.03,
        6: 0.05,
        7: 0.1,
        8: 0.15,
        9: 0.15
    }

    output_0: int = subsample_idx(
        idx=2,
        subsampling_t=1e-2,
        subsampling_pow=0.5,
        idx_to_vocab_freq=idx_to_vocab_freq,
        generate_random_between_0_and_1=lambda: 0.1
    )

    assert output_0 == 2

    output_1: int = subsample_idx(
        idx=2,
        subsampling_t=1e-2,
        subsampling_pow=0.5,
        idx_to_vocab_freq=idx_to_vocab_freq,
        generate_random_between_0_and_1=lambda: 0.2
    )

    assert output_1 == -1

    output_3: int = subsample_idx(
        idx=2,
        subsampling_t=1e-2,
        subsampling_pow=0.8,
        idx_to_vocab_freq=idx_to_vocab_freq,
        generate_random_between_0_and_1=lambda: 0.065
    )

    assert output_3 == 2
