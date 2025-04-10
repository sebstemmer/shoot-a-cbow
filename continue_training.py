import cbow_functions as cf
import torch
from torch import nn
import sys
from torch.utils.data import DataLoader
import pickle


for epoch in range(0, num_epochs):
    print("epoch: " + str(epoch) + "\n")
    for batch in dataloader:
        optimizer.zero_grad()

        x, normed_mask, y = batch

        outputs = model(x.to(device), normed_mask.to(device))

        loss = loss_fn(outputs, y.to(device))

        print("loss: " + str(loss.item()))

        loss.backward()

        optimizer.step()

    checkpoint_path = f"model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
