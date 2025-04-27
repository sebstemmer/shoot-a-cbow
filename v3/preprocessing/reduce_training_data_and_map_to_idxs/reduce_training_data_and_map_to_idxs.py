import v3.preprocessing.preprocessing_logic as preprocessing_logic


def reduce_training_data_and_map_to_idxs(
        full_training_data_in_words: list[tuple[str, list[str]]],
        word_to_idx_count_vocab: dict[str, tuple[int, int]],
) -> list[tuple[int, list[int]]]:
    return [
        (word_to_idx_count_vocab[sample[0]][0], reduced_context_window) for sample in full_training_data_in_words if (
            sample[0] in word_to_idx_count_vocab
        ) and (
            reduced_context_window := preprocessing_logic.reduce_context_window_and_map_to_idx(
                context_window=sample[1],
                word_to_idx_count_vocab=word_to_idx_count_vocab
            )
        )
    ]
