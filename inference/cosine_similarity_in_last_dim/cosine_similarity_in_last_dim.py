import torch


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
