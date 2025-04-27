import torch


def create_word_to_idx_vocab(vocab: dict[str, tuple[int, int]], vocab_size: int) -> dict[str, int]:
    return {
        word: word_count_and_idx[1] for word, word_count_and_idx in vocab.items()
        if word_count_and_idx[1] < vocab_size
    }


def create_idx_to_word_vocab(vocab: dict[str, tuple[int, int]], vocab_size: int) -> dict[int, str]:
    return {
        word_count_and_idx[1]: word for word, word_count_and_idx in vocab.items()
        if word_count_and_idx[1] < vocab_size
    }


def cosine_similarity_in_last_dim(
    vec: torch.Tensor,
    embeddings: torch.Tensor,
    excludes: list[int]
) -> torch.Tensor:
    result: torch.Tensor = torch.nn.functional.cosine_similarity(
        vec,
        embeddings,
        dim=-1
    )

    result[excludes] = 0.0

    return result


def check_relation(
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

    print(sim.shape)

    _, top_idxs_tensor = torch.topk(input=sim, k=top)

    top_idxs: list[int] = top_idxs_tensor.tolist()  # type: ignore

    return [idx_to_word_vocab[idx] for idx in top_idxs]
