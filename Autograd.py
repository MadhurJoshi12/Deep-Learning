import torch 

w = torch.tensor(3.0, requires_grad=False)
# Track this number for learning purpose
x = torch.tensor(4.0)
y = w * x
print(y)
