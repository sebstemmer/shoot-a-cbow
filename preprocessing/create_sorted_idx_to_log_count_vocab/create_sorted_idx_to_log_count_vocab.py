import math


def create_sorted_idx_to_log_count_vocab(
        word_to_idx_count_vocab: dict[str, tuple[int, int]],
        log_safety: float = 1e-10
) -> list[tuple[int, float]]:
    return [
        (
            idx_and_count[0],
            math.log(idx_and_count[1]+log_safety)
        ) for idx_and_count in sorted(
            word_to_idx_count_vocab.values(), key=lambda x: x[1], reverse=True
        )
    ]
