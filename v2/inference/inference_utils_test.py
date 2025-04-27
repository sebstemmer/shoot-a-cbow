import v2.inference.inference_utils as inference_utils
import torch

# create_word_to_idx_vocab

create_word_to_idx_vocab_vocab: dict[str, tuple[int, int]] = {
    "some": (121, 0),
    "word": (12, 1),
    "is": (3, 2),
    "good": (1, 3),
}

create_word_to_idx_vocab_output: dict[str, int] = inference_utils.create_word_to_idx_vocab(
    vocab=create_word_to_idx_vocab_vocab,
    vocab_size=3
)

assert create_word_to_idx_vocab_output == {
    "some": 0,
    "word": 1,
    "is": 2
}


# create_idx_to_word_vocab

create_idx_to_word_vocab_vocab: dict[str, tuple[int, int]] = {
    "some": (121, 0),
    "word": (12, 1),
    "is": (3, 2),
    "good": (1, 3),
}

create_idx_to_word_vocab_output: dict[int, str] = inference_utils.create_idx_to_word_vocab(
    vocab=create_idx_to_word_vocab_vocab,
    vocab_size=3
)

assert create_idx_to_word_vocab_output == {
    0: "some",
    1: "word",
    2: "is"
}


# cosine_similarity_in_last_dim

cosine_similarity_in_last_dim_vec: torch.Tensor = torch.tensor([1, 2, 3])
cosine_similarity_in_last_dim_embeddings: torch.Tensor = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [1.0, 2.0, 3.0],
        [2.0, 2.0, -2.0]
    ]
)

cosine_similarity_in_last_dim_output: torch.Tensor = inference_utils.cosine_similarity_in_last_dim(
    vec=cosine_similarity_in_last_dim_vec.unsqueeze(0),
    embeddings=cosine_similarity_in_last_dim_embeddings,
    excludes=[0, 1]
)

assert torch.allclose(
    cosine_similarity_in_last_dim_output,
    torch.tensor([0.0, 0.0, 1.0, 0.0])
)
