import torch

d1 = torch.tensor([1,2,3,4,5])
print(d1[0])
print(d1[4])

d2 = torch.tensor([
    [10,20,30],
    [40,50,60]
])
print(d2[0,1])
print(d2[1,2])
