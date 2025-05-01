import v3.training.training_utils as training_utils
import matplotlib.pyplot as plt

training_run_label: str = "nosubsampling"

epoch_losses: dict[int, float] = training_utils.load_epoch_losses(
    training_run_label
)

plt.plot(  # type: ignore
    list(epoch_losses.keys()),
    list(epoch_losses.values())
)
plt.xlabel(xlabel="epoch")  # type: ignore
plt.ylabel(ylabel="loss")  # type: ignore
plt.show()  # type: ignore
