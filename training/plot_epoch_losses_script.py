import training.training_utils as training_utils
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

training_run_label: str = "vs_30_cw_4_noss"
toIdx = 70

epoch_losses: dict[int, float] = training_utils.load_epoch_losses(
    training_run_label
)

keys: list[int] = list(epoch_losses.keys())
values: list[float] = list(epoch_losses.values())
fig, ax = plt.subplots()  # type: ignore

ax.plot(  # type: ignore
    list(keys[:toIdx]),
    list(values[:toIdx])
)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.set_xlabel("epoch")  # type: ignore
ax.set_ylabel("loss")  # type: ignore
ax.set_title("Average Loss per Epoch")  # type: ignore
plt.show()  # type: ignore
