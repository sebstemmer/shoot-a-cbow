import torch
import pickle
import preprocessing.preprocessing_utils as preprocessing_utils
import training.training_utils as training_utils


preprocessing_run_label: str = "vs_30_cw_4"
training_run_label: str = "vs_30_cw_4_noss"
load_from_epoch: int = -1

batch_size: int = 2048
hidden_layer_size: int = 300

num_epochs: int = 1000
learning_rate: int = 5

activate_subsampling: bool = False
subsampling_t: float = 1e-4
subsampling_pow: float = 1.0


print("start training-run with label " + training_run_label + "...")


print("load preprocessed-data...")

preprocessed_data: preprocessing_utils.PreprocessedData = training_utils.load_preprocessed_data(
    preprocessing_run_label=preprocessing_run_label
)

print("...preprocessed-data loaded")


print("load vocab...")

vocab: preprocessing_utils.Vocab = training_utils.load_vocab(
    preprocessing_run_label=preprocessing_run_label
)

print("...vocab loaded")


print("there are " + str(len(preprocessed_data.training_data) / batch_size) + " batches")


device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"
print("using device: " + device)


print("init model, optimizer and loss-function...")

model: training_utils.CBOWNeuralNetwork = training_utils.CBOWNeuralNetwork(
    vocab_size=vocab.vocab_size,
    hidden_layer_size=hidden_layer_size
).to(device)

optimizer: torch.optim.SGD = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)

cross_entropy_loss_function: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

print("...inited model, optimizer and loss-function")


if (load_from_epoch >= 0):
    print("load model " + str(training_run_label) +
          " from epoch " + str(load_from_epoch) + "...")

    model_data_at_epoch: training_utils.ModelData = training_utils.load_model_data_at_epoch(
        training_run_label=training_run_label,
        epoch=load_from_epoch
    )

    hidden_layer_size = model_data_at_epoch.hidden_layer_size

    model.load_state_dict(model_data_at_epoch.model_state_dict)

    print("...loaded model")


model.train()


for epoch in range(load_from_epoch + 1, num_epochs):
    print("\n\nepoch: " + str(epoch))

    subsampled_training_data: list[tuple[int, list[int]]]

    if activate_subsampling:
        subsampled_training_data = preprocessing_utils.subsample_training_data(
            training_data=preprocessed_data.training_data,
            idx_to_vocab_freq=vocab.idx_to_vocab_freq,
            subsampling_t=subsampling_t,
            subsampling_pow=subsampling_pow
        )

        print("there are " +
              str(len(subsampled_training_data) / batch_size) + " batches after subsampling")
    else:
        subsampled_training_data = preprocessed_data.training_data

    dataset: training_utils.Dataset = training_utils.Dataset(
        training_data=subsampled_training_data,
        context_window_size=preprocessed_data.context_window_size,
        vocab_size=vocab.vocab_size
    )

    dataLoader: torch.utils.data.DataLoader[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    epoch_loss = 0.0
    counter = 0.0

    for (idx, batch) in enumerate(dataLoader):
        print(idx)
        optimizer.zero_grad()

        x, normed_mask, y = batch

        outputs = model(
            x=x.to(device),
            normed_mask=normed_mask.to(device)
        )

        loss = cross_entropy_loss_function(
            input=outputs,
            target=y.to(device)
        )

        epoch_loss += loss.item()
        counter += 1.0

        loss.backward()

        optimizer.step()  # type: ignore

    epoch_loss = epoch_loss / counter

    print("epoch loss is " + str(epoch_loss))

    print("save model...")

    path_to_model_at_epoch: str = training_utils.path_to_model_at_epoch(
        epoch=epoch,
        training_run_label=training_run_label
    )
    torch.save(  # type: ignore
        training_utils.ModelData(
            epoch=epoch,
            hidden_layer_size=hidden_layer_size,
            model_state_dict=model.state_dict()
        ),
        path_to_model_at_epoch
    )
    print("...model saved")

    print("save epoch-loss...")

    training_utils.save_epoch_loss(
        training_run_label=training_run_label,
        epoch=epoch,
        epoch_loss=epoch_loss
    )

    print("...epoch-loss saved")
