import torch


class CBOWNeuralNetwork(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_layer_size: int):
        super().__init__()
        self.input_to_hidden = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_layer_size
        )
        self.hidden_to_output = torch.nn.Linear(
            in_features=hidden_layer_size,
            out_features=vocab_size,
            bias=False
        )

    def forward(self, x, mask):
        hidden = self.input_to_hidden(x)

        masked_hidden = torch.matmul(mask, hidden)

        output = self.hidden_to_output(masked_hidden.squeeze(1))

        return output
