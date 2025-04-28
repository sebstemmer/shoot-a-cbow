import v3.preprocessing.preprocessing_utils as preprocessing_utils
import matplotlib.pyplot as plt
import pickle
import copy

# 1e-4 0.9

subsampling_t = 1e-4
subsampling_pow = 1


print("load preprocessed-data...")

with open(preprocessing_utils.path_to_data_folder + "preprocessing/preprocessed_data.pickle", "rb") as handle:
    preprocessed_data: preprocessing_utils.PreprocessedData = pickle.load(
        handle
    )

print("...preprocessed-data loaded")


print("load vocab...")

with open(preprocessing_utils.path_to_data_folder + "preprocessing/vocab.pickle", "rb") as handle:
    vocab: preprocessing_utils.Vocab = pickle.load(handle)

print("...vocab loaded")


print("create idx-and-log-count-sorted...")

idx_and_log_count_sorted: list[tuple[int, float]] = preprocessing_utils.create_sorted_idx_to_log_count_vocab(
    word_to_idx_count_vocab=vocab.word_to_idx_count_vocab
)

print("...idx-and-log-count-sorted created")


print("subsample training-data...")

subsampled_training_data: list[tuple[int, list[int]]] = preprocessing_utils.subsample_training_data(
    training_data=preprocessed_data.training_data,
    idx_to_vocab_freq=vocab.idx_to_vocab_freq,
    subsampling_t=subsampling_t,
    subsampling_pow=subsampling_pow
)

print("...subsampled training-data")


print("create subsampled-word-to-idx-count-vocab...")

subsampled_word_to_idx_count_vocab: dict[str, tuple[int, int]] = copy.deepcopy(
    vocab.word_to_idx_count_vocab
)

subsampled_idx_to_word_vocab: dict[int, str] = preprocessing_utils.create_idx_to_word_vocab(
    word_to_idx_count_vocab=subsampled_word_to_idx_count_vocab
)

subsampled_word_to_idx_count_vocab: dict[str, tuple[int, int]] = preprocessing_utils.add_window_count_to_word_to_idx_count_vocab(
    word_to_idx_count_vocab=subsampled_word_to_idx_count_vocab,
    idx_to_word_vocab=subsampled_idx_to_word_vocab,
    training_data=subsampled_training_data
)

print("...subsampled-word-to-idx-count-vocab created")


print("len(subsampled_training_data)")
print(len(subsampled_training_data))


print("create subsampled-idx-and-log-count-sorted...")

subsampled_idx_and_log_count_sorted: list[tuple[int, float]] = preprocessing_utils.create_sorted_idx_to_log_count_vocab(
    word_to_idx_count_vocab=subsampled_word_to_idx_count_vocab
)

print("...subsampled-idx-and-log-count-sorted created")


""" subsampled_idx_to_count_vocab: dict[int, int] = {
    subsampled_idx_count[0]: subsampled_idx_count[1] for subsampled_idx_count in subsampled_word_to_idx_count_vocab.values()
}

sorted_subsampled_values: list[tuple[int, int]] = [
    (a[0], subsampled_idx_to_count_vocab[a[0]]) for a in idx_count_sorted
]

sorted(subsampled_word_to_idx_count_vocab.values(),
       key=lambda x: x[1], reverse=True)


count_subsampled = [(math.log(v[1]) if v[1] > 0 else 0)
                    for v in sorted_subsampled_values] """

plt.plot(  # type: ignore
    range(0, len(idx_and_log_count_sorted)),
    list(map(lambda x: x[1], idx_and_log_count_sorted))
)
plt.plot(  # type: ignore
    range(0, len(subsampled_idx_and_log_count_sorted)),
    list(map(lambda x: x[1], subsampled_idx_and_log_count_sorted))
)
plt.show()  # type: ignore
