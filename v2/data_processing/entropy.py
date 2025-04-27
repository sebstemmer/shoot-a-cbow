import pickle
import v2.training.training_utils as training_utils
import v2.inference.inference_utils as inference_utils

with open("./v2/data/processing/processed_data.pickle", "rb") as handle:
    processed_data = pickle.load(handle)

with open("./v2/data/processing/vocab_data.pickle", "rb") as handle:
    vocab_data = pickle.load(handle)


vocab: dict[str, tuple[int, int]] = vocab_data["vocab"]

vocab_frequencies: dict[int, float] = training_utils.create_vocab_frequencies(
    vocab
)

context_window_size: int = processed_data["context_window_size"]
vocab_size: int = processed_data["vocab_size"]
training_data: list[tuple[int, list[int]]] = processed_data["training_data"]

# calc entropy

print("calc entropy...")

ent_N = torch.zeros(vocab_size)
ent_nlogn = torch.zeros(vocab_size)

for td_idx, td in enumerate(training_data[0:1000000]):
    print(td_idx)
    all: list[int] = [td[0]] + td[1]

    # u = (torch.tensor(all) == torch.arange(  # type: ignore
    #    0, vocab_size).unsqueeze(1)).sum(dim=1)

    u = torch.bincount(torch.tensor(all), minlength=vocab_size)

    # torch.where(u > 0, u * torch.log((u+1e-10)), 0.0)
    ent_nlogn += u * torch.log((u+1e-10))
    ent_N += u

ent_N_save = torch.where(ent_N > 0, ent_N, 0)

ent = torch.where(
    ent_N_save > 0,
    -1.0 / ent_N_save * (
        ent_nlogn - torch.where(
            ent_N_save > 0, torch.log(
                ent_N_save
            ), 0.0) * ent_N_save
    ), 0.0)

top_values, top_indices = torch.topk(ent_N_save, k=30000)

# plt.hist(top_values)  # type: ignore
# plt.show()  # type: ignore

print(top_values[0:100])
print(top_indices[0:100])

print(top_values[2500:2600])
print(top_indices[2500:2600])

plt.plot(range(0, 30000), top_values)  # type: ignore
plt.show()  # type: ignore

idx_to_word_vocab = inference_utils.create_idx_to_word_vocab(
    vocab=vocab, vocab_size=vocab_size)

print([idx_to_word_vocab[top_val]
      for top_val in top_indices.tolist()])  # type: ignore
