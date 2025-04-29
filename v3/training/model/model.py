import torch


class CBOWNeuralNetwork(torch.nn.Module):
    def __init__(
            self: torch.nn.Module,
            vocab_size: int,
            hidden_layer_size: int
    ):
        super().__init__()
        self.input_to_hidden: torch.nn.Embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_layer_size
        )
        self.hidden_to_output: torch.nn.Linear = torch.nn.Linear(
            in_features=hidden_layer_size,
            out_features=vocab_size,
            bias=False
        )

    def forward(self: 'CBOWNeuralNetwork', x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.input_to_hidden(x)

        masked_hidden = torch.matmul(mask, hidden)

        return self.hidden_to_output(masked_hidden.squeeze(1))
