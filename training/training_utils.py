from .model.model import *
from .data_set.data_set import *
import preprocessing.preprocessing_utils as preprocessing_utils
import os
import pickle
from typing import Any


def load_vocab(preprocessing_run_label: str) -> preprocessing_utils.Vocab:
    with open(preprocessing_utils.path_to_vocab(
        preprocessing_run_label=preprocessing_run_label
    ), "rb") as handle:
        vocab: preprocessing_utils.Vocab = pickle.load(handle)
        return vocab


def load_preprocessed_data(preprocessing_run_label: str) -> preprocessing_utils.PreprocessedData:
    with open(preprocessing_utils.path_to_preprocessed_data(
        preprocessing_run_label=preprocessing_run_label
    ), "rb") as handle:
        preprocessed_data: preprocessing_utils.PreprocessedData = pickle.load(
            handle
        )
        return preprocessed_data


path_to_training_folder: str = preprocessing_utils.path_to_data_folder + "training/"


def path_to_training_run_folder(training_run_label: str) -> str:
    path = path_to_training_folder + str(training_run_label)
    os.makedirs(path, exist_ok=True)
    return path + "/"


def path_to_model_folder(training_run_label: str) -> str:
    path = path_to_training_run_folder(
        training_run_label=training_run_label
    ) + "model/"
    os.makedirs(path, exist_ok=True)
    return path


def path_to_model_at_epoch(epoch: int, training_run_label: str) -> str:
    return path_to_model_folder(training_run_label=training_run_label) + "epoch_" + str(epoch) + ".pt"


class ModelData:
    def __init__(
        self,
        epoch: int,
        model_state_dict: dict[str, Any],
        hidden_layer_size: int
    ):
        self.epoch: int = epoch
        self.model_state_dict: dict[str, Any] = model_state_dict
        self.hidden_layer_size: int = hidden_layer_size


def load_model_data_at_epoch(
    training_run_label: str,
    epoch: int
) -> ModelData:
    model_data_at_epoch: ModelData = torch.load(  # type: ignore
        path_to_model_at_epoch(
            epoch=epoch,
            training_run_label=training_run_label
        ),
        weights_only=False
    )

    return model_data_at_epoch


def load_epoch_losses(
    training_run_label: str
) -> dict[int, float]:
    with open(
        path_to_training_run_folder(
            training_run_label=training_run_label
        ) + "epoch_losses.pickle", "rb"
    ) as handle:
        epoch_losses: dict[int, float] = pickle.load(
            handle
        )
        return epoch_losses


def save_epoch_loss(
        training_run_label: str,
        epoch: int,
        epoch_loss: float
):
    if epoch > 0:
        epoch_losses: dict[int, float] = load_epoch_losses(training_run_label)
        epoch_losses[epoch] = epoch_loss
    else:
        epoch_losses: dict[int, float] = {0: epoch_loss}

    with open(
        path_to_training_run_folder(
            training_run_label=training_run_label
        ) + "epoch_losses.pickle", "wb"
    ) as handle:
        pickle.dump(
            epoch_losses,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL
        )
