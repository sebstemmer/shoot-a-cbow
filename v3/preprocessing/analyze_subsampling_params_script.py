from v3.preprocessing.path_to_data_folder_const import path_to_data_folder
from v3.preprocessing.vocab_class import Vocab
from v3.preprocessing.preprocessed_data_class import PreprocessedData
import v3.preprocessing.preprocessing_utils as preprocessing_utils
import matplotlib.pyplot as plt
import pickle
import math
import v3.training.training_utils as training_utils
import copy

# 1e-4 0.9

subsampling_t = 1e-4
subsampling_pow = 1

with open(path_to_data_folder + "preprocessing/preprocessed_data.pickle", "rb") as handle:
    preprocessed_data: PreprocessedData = pickle.load(
        handle
    )

with open(path_to_data_folder + "preprocessing/vocab.pickle", "rb") as handle:
    vocab: Vocab = pickle.load(handle)

# idx, count
sorted_values: list[tuple[int, int]] = sorted(
    vocab.word_to_idx_count_vocab.values(),
    key=lambda x: x[1],
    reverse=True
)

count = [math.log(v[1]) for v in sorted_values]

print(sorted_values[1200])


vocab_frequencies: dict[int, float] = training_utils.create_vocab_frequencies(
    word_to_idx_count_vocab=vocab.word_to_idx_count_vocab
)

print(vocab_frequencies[1209])

print("subsample")

subsampled_training_data: list[tuple[int, list[int]]] = training_utils.subsample_training_data(
    training_data=preprocessed_data.training_data,
    vocab_frequencies=vocab_frequencies,
    subsampling_t=subsampling_t,
    subsampling_pow=subsampling_pow
)

subsampled_word_to_idx_count_vocab: dict[str, tuple[int, int]] = copy.deepcopy(
    vocab.word_to_idx_count_vocab
)

subsampled_idx_to_word_vocab = preprocessing_utils.create_idx_to_word_vocab(
    word_to_idx_count_vocab=subsampled_word_to_idx_count_vocab
)

print("count")

subsampled_word_to_idx_count_vocab: dict[str, tuple[int, int]] = preprocessing_utils.add_window_count_to_vocab(
    word_to_idx_count_vocab=subsampled_word_to_idx_count_vocab,
    idx_to_word_vocab=subsampled_idx_to_word_vocab,
    training_data=subsampled_training_data
)

print("len(subsampled_training_data)")
print(len(subsampled_training_data))

subsampled_idx_to_count_vocab: dict[int, int] = {
    subsampled_idx_count[0]: subsampled_idx_count[1] for subsampled_idx_count in subsampled_word_to_idx_count_vocab.values()
}

sorted_subsampled_values: list[tuple[int, int]] = [
    (a[0], subsampled_idx_to_count_vocab[a[0]]) for a in sorted_values
]

sorted(subsampled_word_to_idx_count_vocab.values(),
       key=lambda x: x[1], reverse=True)


count_subsampled = [(math.log(v[1]) if v[1] > 0 else 0)
                    for v in sorted_subsampled_values]

plt.plot(range(0, len(sorted_values)), count)
plt.plot(range(0, len(sorted_subsampled_values)), count_subsampled)
plt.show()
