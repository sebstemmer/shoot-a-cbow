from .cosine_similarity_in_last_dim.cosine_similarity_in_last_dim import *


def calc_embedding_a_minus_b_plus_c(
    word_a: str,
    word_b: str,
    word_c: str,
    embeddings: torch.Tensor,
    word_to_idx_vocab: dict[str, int],
    idx_to_word_vocab: dict[int, str],
    top: int
) -> list[str]:
    word_a_idx: int = word_to_idx_vocab[word_a]
    word_a_embed: torch.Tensor = embeddings[word_a_idx, :]

    word_b_idx: int = word_to_idx_vocab[word_b]
    word_b_embed: torch.Tensor = embeddings[word_b_idx, :]

    word_c_idx: int = word_to_idx_vocab[word_c]
    word_c_embed: torch.Tensor = embeddings[word_c_idx, :]

    result: torch.Tensor = word_a_embed - word_b_embed + word_c_embed

    sim: torch.Tensor = cosine_similarity_in_last_dim(
        vec=result,
        embeddings=embeddings,
        excludes=[word_a_idx, word_b_idx, word_c_idx]
    )

    _, top_idxs_tensor = torch.topk(input=sim, k=top)

    top_idxs: list[int] = top_idxs_tensor.tolist()  # type: ignore

    return [idx_to_word_vocab[idx] for idx in top_idxs]
