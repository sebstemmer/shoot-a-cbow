import torch
from torch import nn

num_training_samples = 3
a_1_dim = 5

a = torch.rand(num_training_samples, a_1_dim, 7)

print("a")
print(a)

b = torch.tensor([1, 5, 2])

print("b")
print(b)

mean = torch.zeros(
    num_training_samples, a.shape[2]
)
for idx, bEl in enumerate(b):
    mean[idx] = torch.mean(a[idx, 0:bEl.item(), :], dim=0)

print("mean.shape")
print(mean.shape)

print("mean")
print(mean)

b_dim = b.unsqueeze(1)

c = torch.arange(0, a_1_dim).unsqueeze(0)

print(c)

d = (c < b_dim).float()
d_sums = d.sum(dim=1, keepdim=True)

print(d)
print(d_sums)

print("d")
print(d)

normed_d = d/d_sums

normed_d_unsqueeze_1 = normed_d.unsqueeze(1)

print("normed_d")
print(normed_d)

print("normed_d.shape()")
print(normed_d.shape)

print("a.shape")
print(a.shape)

print("normed_d_unsqueeze_1.shape")
print(normed_d_unsqueeze_1.shape)

f = torch.matmul(normed_d_unsqueeze_1, a)

print("f.shape")
print(f.shape)

print("f")
print(f.squeeze(1))

# print(d.shape)
# print(a.shape)

# print(f)
