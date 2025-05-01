import torch


class Dataset(torch.utils.data.Dataset[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]):
    def __init__(
            self,
            training_data:  list[tuple[int, list[int]]],
            context_window_size: int,
            vocab_size: int
    ):
        self.training_data = training_data
        self.context_window_size = context_window_size
        self.vocab_size = vocab_size

    def __len__(self: 'Dataset'):
        return len(self.training_data)

    def __getitem__(self: 'Dataset', idx: int):
        training_sample: tuple[int, list[int]] = self.training_data[idx]

        X: torch.Tensor = torch.full(
            [self.context_window_size * 2],
            fill_value=self.vocab_size
        )
        X[0:len(training_sample[1])] = torch.tensor(training_sample[1])

        CS: torch.Tensor = torch.tensor(
            len(training_sample[1]), dtype=torch.short)

        mask: torch.Tensor = (
            torch.arange(0, self.context_window_size * 2) < CS
        ).float()

        normed_mask = mask / mask.sum()
        normed_mask = normed_mask.unsqueeze(0)

        Y: torch.Tensor = torch.tensor(training_sample[0])

        return X, normed_mask, Y
