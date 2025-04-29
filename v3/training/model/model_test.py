import torch
from .model import CBOWNeuralNetwork


def runTest():
    vocab_size: int = 5
    hidden_layer_size: int = 3

    x: torch.Tensor = torch.tensor(
        [
            [2, 1, 5],
            [3, 0, 1]
        ]
    )

    mask: torch.Tensor = torch.tensor(
        [
            [[0.5, 0.5, 0.0]],
            [[1.0/3.0, 1.0/3.0, 1.0/3.0]]
        ]
    )

    model: CBOWNeuralNetwork = CBOWNeuralNetwork(
        vocab_size=vocab_size,
        hidden_layer_size=hidden_layer_size
    )

    model.input_to_hidden.weight.data = torch.tensor([
        [1.0, 6.0, 11.0],
        [2.0, 7.0, 12.0],
        [3.0, 8.0, 13.0],
        [4.0, 9.0, 14.0],
        [5.0, 10.0, 15.0],
        [0.0, 0.0, 0.0]
    ])

    model.hidden_to_output.weight.data = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ])

    output = model.forward(
        x=x,
        mask=mask
    )

    assert torch.allclose(
        output,
        torch.tensor(
            [
                [55.0000, 122.5000, 190.0000, 257.5000, 325.0000],
                [54.0000, 120.0000, 186.0000, 252.0000, 318.0000]
            ]
        )
    )
