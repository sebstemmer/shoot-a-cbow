import training.training_utils as training_utils
import matplotlib.pyplot as plt

training_run_label_0: str = "vs_30_cw_4_noss"

epoch_losses_0: dict[int, float] = training_utils.load_epoch_losses(
    training_run_label_0
)

training_run_label_1: str = "vs_30_cw_6_noss"

epoch_losses_1: dict[int, float] = training_utils.load_epoch_losses(
    training_run_label_1
)

plt.plot(  # type: ignore
    list(epoch_losses_0.keys()),
    list(epoch_losses_0.values())
)
plt.plot(  # type: ignore
    list(epoch_losses_1.keys()),
    list(epoch_losses_1.values())
)
plt.xlabel(xlabel="epoch")  # type: ignore
plt.ylabel(ylabel="loss")  # type: ignore
plt.show()  # type: ignore
