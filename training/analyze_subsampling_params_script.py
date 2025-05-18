import preprocessing.preprocessing_utils as preprocessing_utils
import matplotlib.pyplot as plt
import pickle
import copy

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


plt.plot(  # type: ignore
    range(0, len(idx_and_log_count_sorted)),
    list(map(lambda x: x[1], idx_and_log_count_sorted))
)


def create_plot_for_subsample_params(
    subsampling_t: float,
    subsampling_pow: float,
    legend_label: str
):
    print("\n\ncreate plot for " + str(legend_label))

    print("subsample training-data...")

    subsampled_training_data: list[tuple[int, list[int]]] = preprocessing_utils.subsample_training_data(
        training_data=preprocessed_data.training_data,
        idx_to_vocab_freq=vocab.idx_to_vocab_freq,
        subsampling_t=subsampling_t,
        subsampling_pow=subsampling_pow
    )

    print("...subsampled training-data")

    print("there are " + str(len(subsampled_training_data)) +
          " subsampled training samples")

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

    print("create subsampled-idx-and-log-count-sorted...")

    subsampled_idx_and_log_count_sorted: list[tuple[int, float]] = preprocessing_utils.create_sorted_idx_to_log_count_vocab(
        word_to_idx_count_vocab=subsampled_word_to_idx_count_vocab
    )

    print("...subsampled-idx-and-log-count-sorted created")

    r, = plt.plot(  # type: ignore
        range(0, len(subsampled_idx_and_log_count_sorted)),
        list(map(lambda x: x[1], subsampled_idx_and_log_count_sorted)),
        label=legend_label
    )


create_plot_for_subsample_params(
    subsampling_t=1e-4,
    subsampling_pow=0.9,
    legend_label="1e-4, 0.9"
)

create_plot_for_subsample_params(
    subsampling_t=1e-4,
    subsampling_pow=1,
    legend_label="1-e4, 1"
)

create_plot_for_subsample_params(
    subsampling_t=1e-2,
    subsampling_pow=0.5,
    legend_label="1-e2, 0.5"
)

create_plot_for_subsample_params(
    subsampling_t=1e-2,
    subsampling_pow=1,
    legend_label="1-e2, 1"
)

create_plot_for_subsample_params(
    subsampling_t=1e-3,
    subsampling_pow=1,
    legend_label="1-e3, 1"
)

create_plot_for_subsample_params(
    subsampling_t=1e-5,
    subsampling_pow=0.9,
    legend_label="1-e5, 0.9"
)

create_plot_for_subsample_params(
    subsampling_t=1e-5,
    subsampling_pow=1,
    legend_label="1e-5, 1"
)

plt.legend()  # type: ignore
plt.show()  # type: ignore
