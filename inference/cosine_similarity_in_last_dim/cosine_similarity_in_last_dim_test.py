from .cosine_similarity_in_last_dim import cosine_similarity_in_last_dim
import torch


def runTest():
    vec: torch.Tensor = torch.tensor([1, 2, 3])
    embeddings: torch.Tensor = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0],
            [2.0, 2.0, -2.0]
        ]
    )

    output: torch.Tensor = cosine_similarity_in_last_dim(
        vec=vec.unsqueeze(0),
        embeddings=embeddings,
        excludes=[0, 1]
    )

    assert torch.allclose(
        output,
        torch.tensor([0.0, 0.0, 1.0, 0.0])
    )
